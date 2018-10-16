import cv2





capture = cv2.VideoCapture(0)

# # 定义编码方式并创建VideoWriter对象
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# outfile = cv2.VideoWriter('output.avi', fourcc, 25., (640, 480))
#
# while(capture.isOpened()):
#     ret, frame = capture.read()
#
#     if ret:
#         outfile.write(frame)  # 写入文件
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) == ord('q'):
#             break
#     else:
#         break

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)
while cap.isOpened():
    _, frame = cap.read()
    if _:
        cv2.imwrite("123.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        break


print(capture.get(3))
cap.release()

