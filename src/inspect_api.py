from livekit import api
import sys

with open("fields_info.txt", "w") as f:
    f.write("FIELDS IN CreateSIPParticipantRequest:\n")
    for field in api.CreateSIPParticipantRequest.DESCRIPTOR.fields:
        f.write(f" - {field.name}\n")
    
    f.write("\nDIR(api):\n")
    f.write(str(dir(api)))
