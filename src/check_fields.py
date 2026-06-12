from livekit import api
import inspect

fields = api.CreateSIPParticipantRequest().DESCRIPTOR.fields_by_name.keys()
print("Available fields in CreateSIPParticipantRequest:")
for f in fields:
    print(f" - {f}")
