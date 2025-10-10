import cv2
import requests
import json

# Replace with your camera info
ip = "192.168.50.138"
user = "admin"
password = "r1project"

# Reolink common RTSP stream (sub-stream = lower resolution, faster)
rtsp_url = f"rtsp://{user}:{password}@{ip}:554/Preview_01_sub"

cap = cv2.VideoCapture(rtsp_url)


def send_cmd(cmd, payload=None):
    url = f"http://{ip}/cgi-bin/api.cgi?cmd={cmd}&user={user}&password={password}"
    if payload:
        r = requests.post(url, data=json.dumps(payload))
    else:
        r = requests.get(url)
    return r.text


# Turn IR night vision LEDs ON
def ir_on():
    payload = [{"cmd":"SetIrLights","action":0,"param":{"IrLights":{"state":"On"}}}]
    print(send_cmd("SetIrLights", payload))

def ir_off():
    payload = [{"cmd":"SetIrLights","action":0,"param":{"IrLights":{"state":"Off"}}}]
    print(send_cmd("SetIrLights", payload))


if not cap.isOpened():
    print("❌ Failed to open stream")
    exit()
ir_on()  # Turn on IR night vision LEDs
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Reolink Stream", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# from onvif import ONVIFCamera

# ip = "192.168.50.138"   # your camera IP
# user = "admin"
# password = "r1project"

# cam = ONVIFCamera(ip, 8000, user, password)  # Reolink ONVIF port is usually 8000
# media = cam.create_media_service()

# profiles = media.GetProfiles()
# for i, p in enumerate(profiles):
#     enc = p.VideoEncoderConfiguration
#     res = (enc.Resolution.Width, enc.Resolution.Height)
#     fps = enc.RateControl.FrameRateLimit if enc.RateControl else "?"
#     print(f"[{i}] {p.Name}  {enc.Encoding} {res} {fps}fps  Token={p.token}")

#     req = media.create_type('GetStreamUri')
#     req.StreamSetup = {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}
#     req.ProfileToken = p.token
#     uri = media.GetStreamUri(req).Uri
#     print("   URI:", uri)