import requests
import json

# Camera info
IP = "192.168.50.138"
USER = "admin"
PASS = "r1project"

def send_cmd(cmd, payload=None):
    url = f"http://{IP}/cgi-bin/api.cgi?cmd={cmd}&user={USER}&password={PASS}"
    if payload:
        r = requests.post(url, data=json.dumps(payload))
    else:
        r = requests.get(url)
    return r.text

# --- Examples ---

# Turn IR night vision LEDs ON
def ir_on():
    payload = [{"cmd":"SetIrLights","action":0,"param":{"IrLights":{"state":"On"}}}]
    print(send_cmd("SetIrLights", payload))

# Turn IR night vision LEDs OFF
def ir_off():
    payload = [{"cmd":"SetIrLights","action":0,"param":{"IrLights":{"state":"Off"}}}]
    print(send_cmd("SetIrLights", payload))

# Turn White LED floodlight ON
def led_on(level=None, channel=0):
    """Turn spotlight ON. level can be 1..100 (model-dependent)."""
    p = {"WhiteLed": {"state": "On", "channel": channel}}
    if level is not None:  # many models accept brightness level
        p["WhiteLed"]["level"] = int(level)
    payload = [{"cmd": "SetWhiteLed", "action": 0, "param": p}]
    return send_cmd("SetWhiteLed", payload)

# Turn White LED floodlight OFF
def led_off():
    payload = [{"cmd":"SetWhiteLed","action":0,"param":{"WhiteLed":{"state":"Off"}}}]
    print(send_cmd("SetWhiteLed", payload))

if __name__ == "__main__":
    # Example usage
    # ir_on()
    led_on()
    ir_off()
    # led_off()
