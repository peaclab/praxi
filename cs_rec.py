from time import time
from deltasherlock.common import dictionaries, changesets, fingerprinting, io
from deltasherlock.client import ds_watchdog

#cs = changesets.Changeset(open_time = time())

watch_paths = ["/home/ubuntu/praxi/"]
dswd = ds_watchdog.DeltaSherlockWatchdog(watch_paths, "*", ".")
# Recording begins immediately after instantiation.
# Some time passes... an app installation occurs
input("Press Enter to continue")
# Save changeset
cs = dswd.mark()

repr(cs)

print("encoding cs")
encoded = io.DSEncoder().encode(cs)
repr(encoded)
# Out: '{"open": false, "type": "Changeset", "labe ... cord", "neighbors": [], "filename": "/var/test3"}]}'
# Notice how the output is just a regular old JSON string, and can easily be copy/pasted or saved to file
"""decoded = io.DSDecoder().decode(encoded)
repr(decoded == cs)
# Out: 'True'
io.save_object_as_json(cs, "/home/ubuntu/dummy_cs.dscs")

io.save_object_as_json(dummy_cs, "/home/ubuntu/praxi/dummy_cs.dscs")

print("File end")
# Save changeset"""
