
# # from database import db_utils, db_setup
# # print("Classrooms:", db_utils.get_classrooms())
# # cid = 2
# # seats = db_utils.get_seats_by_classroom(cid)
# # print("Seats for classroom", cid, ":")
# # for s in seats:
# #     print(s)
# # print("Number of seats:", len(seats))


# # from database import db_utils
# # cid = 2
# # logs = db_utils.get_verification_logs_for_classroom(cid, limit=50)
# # print("Logs for classroom", cid, " (latest):", len(logs))
# # for l in logs[:10]:
# #     print(l)


# # import os, glob
# # files = sorted(glob.glob("assets/captured_faces/*"), key=os.path.getmtime, reverse=True)[:20]
# # print("Recent captured faces:", len(files))
# # for f in files[:10]:
# #     print(f, os.path.getmtime(f))

# from database import db_setup
# from database.db_setup import VerificationLog
# print("VerificationLog columns:", [c.name for c in VerificationLog.__table__.columns])



# # debug_show_crops.py
# import glob, os, cv2
# files = sorted(glob.glob("assets/captured_faces/*"), key=os.path.getmtime, reverse=True)[:10]
# print("Showing recent captured files:", len(files))
# for f in files:
#     print(f)
#     img = cv2.imread(f)
#     if img is None:
#         print("  cannot read file")
#         continue
#     cv2.imshow(os.path.basename(f), cv2.resize(img, (640, int(640*img.shape[0]/img.shape[1]))))
#     key = cv2.waitKey(0)  # press any key to go to next
#     cv2.destroyAllWindows()


# # dev_expand_bbox.py
# import json, os
# from database import db_utils

# cid = 2
# seats = db_utils.get_seats_by_classroom(cid)
# for s in seats:
#     bbox = s['bbox']
#     x,y,w,h = bbox
#     # expand by 50% around center
#     cx, cy = x + w/2, y + h/2
#     neww, newh = int(w * 1.6), int(h * 1.6)
#     newx = max(0, int(cx - neww/2))
#     newy = max(0, int(cy - newh/2))
#     newbbox = [newx, newy, neww, newh]
#     db_utils.insert_or_update_seat(cid, s['seat_label'], json.dumps(newbbox), s['row'], s['col'], s['assigned_roll_no'])
#     print("Updated", s['seat_label'], "->", newbbox)
# print("Done")


# debug_check_faces.py
import glob, cv2, os
HAAR = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
fc = cv2.CascadeClassifier(HAAR)

files = sorted(glob.glob("assets/captured_faces/*"), key=os.path.getmtime, reverse=True)[:30]
print("Checking", len(files), "recent captures")
for f in files:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
    print(f"{os.path.basename(f)} -> faces detected: {len(faces)}")
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    # save annotated to debug folder
    outdir = "debug/annotated"
    import os
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, os.path.basename(f)), img)
print("Annotated images written to debug/annotated/")
