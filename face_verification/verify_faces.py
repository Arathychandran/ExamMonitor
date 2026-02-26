# from deepface import DeepFace

# def verify(student_img, live_img):
#     result = DeepFace.verify(
#         img1_path=student_img,
#         img2_path=live_img,
#         model_name="VGG-Face",
#         enforce_detection=False
#     )
#     return result


# face_verification/verify_faces.py
from deepface import DeepFace
def verify(student_img, captured_img, model_name="VGG-Face", enforce_detection=False):
    try:
        res = DeepFace.verify(img1_path=student_img, img2_path=captured_img, model_name=model_name, enforce_detection=enforce_detection)
        # res contains 'verified' boolean and 'distance' numeric
        return res
    except Exception as e:
        return {"verified": False, "error": str(e)}
