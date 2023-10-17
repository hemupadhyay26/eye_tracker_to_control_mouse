import cv2 as cv
import mediapipe as mp
import pyautogui as pag

cam = cv.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pag.size()
while True:
    _, frame = cam.read()
    frame = cv.flip(frame,1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    # print(landmark_points)
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv.circle(frame,(x,y),3,(0,255,0))
            if id == 1:
                screen_x = screen_w/frame_w *x
                screen_y = screen_h/frame_h *y
                pag.moveTo(screen_x,screen_y)
        left_eye = [landmarks[145],landmarks[159]]
        for landmark in left_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv.circle(frame, (x, y), 3, (0, 0, 255))
        if(left_eye[0].y - left_eye[1].y)<0.004:
            pag.click()
            pag.sleep(1)
    cv.imshow("eye controller", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cam.release()
cv.destroyAllWindows()