from flask import Flask, request
import json
import requests
import process

app = Flask(__name__)

# Page Access Token
PAT = 'EAAVEybpPrIYBADEdXt2hHNPZAx35y5GhHnpiUXw4We0QkbBb2Wl0VJu9Kj1rdM1zs7NCFqKSAMTaszJWZAEjmweBW9Q4iZC1eotnvZCG2YLTWzJoqUP1bYmzgt9GXe2mV9tzZCbWZC4kCAZCf1wdZCzlZCD0Q6ON5wbpIgOGqvnDTNwZDZD'

@app.route('/', methods=['GET'])
def handle_verification():
  # used to verify when making Facebook page
  print("Handling Verification.")
  if request.args.get('hub.verify_token', '') == 'brobot_verify_me_token':
    print("Verification successful!")
    return request.args.get('hub.challenge', '')
  else:
    print("Verification failed!")
    return 'Error, wrong validation token'

@app.route('/', methods=['POST'])
def handle_messages():
  print("Handling Messages")
  payload = request.get_data()
  print(payload)
  for sender, message in messaging_events(payload):
    print("Incoming from %s: %s" % (sender, message))
    send_message(PAT, sender, message)
  return "ok"

def messaging_events(payload):
  """Get tuples of (sender_id, message_text) from
  the data.
  """
  data = json.loads(payload)
  print(data)
  messaging_events = data["entry"][0]["messaging"]
  for event in messaging_events:
    if "message" in event and "text" in event["message"]:
      yield event["sender"]["id"], event["message"]["text"].encode('unicode_escape')
    else:
      # initial translation
      yield event["sender"]["id"], "thumbs up"

# get prediction from text
def send_message(token, recipient, text):
  """Send the message text to recipient with id recipient.
  """
  text = process.get_prediction(text)
  r = requests.post("https://graph.facebook.com/v2.6/me/messages",
    params={"access_token": token},
    data=json.dumps({
      "recipient": {"id": recipient},
      "message": {"text": text.decode('unicode_escape')}
    }),
    headers={'Content-type': 'application/json'})
  if r.status_code != requests.codes.ok:
    print(r.text)

if __name__ == '__main__':
  app.run()
