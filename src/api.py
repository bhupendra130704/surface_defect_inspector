import cv2 as cv
import numpy as np
import sqlite3
import time
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

app = FastAPI(title = 'Metal Defect Inspector API')
model = YOLO('Requirement_files/best.pt')

CONFIDENCE_THRESH = 0.5
CRITICAL_CLASSES = ['inclusion', 'pitted_surface']

ROI_POINTS = np.array([[50,  50],
                        [590, 50],
                        [590, 590],
                        [50,  590]])

@app.get('/health')
def health():
    return {
        'status':  'running',
        'model':   'YOLOv8m NEU Defect Inspector',
        'classes': list(model.names.values())

    }

@app.post('/inspect')
async def inspect(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    frame = cv.resize(frame, (640, 640))

    t0 = time.time()
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESH)
    t1=time.time()
    r = results[0]

    inference_ms = round((t1-t0)*1000,1)

    detections = []
    overall_decision = 'PASS'

    for box in r.boxes:
        conf=float(box.conf[0])
        cls=int(box.cls[0])
        label=model.names[cls]
        x1,y1,x2,y2=map(int,box.xyxy[0])
        cx,cy= (x1+x2)//2, (y1+y2)//2

        inside = cv.pointPolygonTest(ROI_POINTS,(cx,cy), False)

        if inside >= 0:
            if label in CRITICAL_CLASSES and conf > 0.7:
                decision         = 'REJECT'
                overall_decision = 'REJECT'
            else:
                decision = 'FLAG'
                if overall_decision != 'REJECT':
                    overall_decision = 'FLAG'

            detections.append({
                'defect':     label,
                'confidence': round(conf, 3),
                'bbox':       [x1, y1, x2, y2],
                'decision':   decision
            })

            conn = sqlite3.connect('Requirement_files/defect_log.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO inspections
                (timestamp, defect, confidence, decision)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now().isoformat(), label, conf, decision))
            conn.commit()
            conn.close()

    return JSONResponse({
        'detections': detections,
        'overall_decision': overall_decision,
        'total_defects': len(detections),
        'inference_time_ms': inference_ms
    })

@app.get('/stats')
def stats():
    conn = sqlite3.connect('Requirement_files/defect_log.db')
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM inspections")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM inspections WHERE decision='REJECT'")
    rejects = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM inspections WHERE decision='FLAG'")
    flags = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence) FROM inspections")
    avg_conf = round(cursor.fetchone()[0] or 0, 3)

    cursor.execute("""
        SELECT defect, COUNT(*) as count
        FROM inspections
        GROUP BY defect
        ORDER BY count DESC
    """)
    by_defect = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()
    return {
        'total_inspections': total,
        'rejects':           rejects,
        'flags':             flags,
        'pass':              total - rejects - flags,
        'avg_confidence':    avg_conf,
        'defect_rate_%':     round(rejects/total*100, 1) if total > 0 else 0,
        'by_defect_type':    by_defect
    }

