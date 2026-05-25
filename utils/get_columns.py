import fcsparser
import json

path_cal = "./data/Experiment 2026!05!12 14!06/96 Well Plate (standard) - 1/Calcein/D01 Well - D01.fcs"
meta, _ = fcsparser.parse(path_cal)

# To avoid massive output, print key categories and a few specific interesting fields
interesting_keys = ['__header__', '$CYT', '$DATE', '$BTIM', '$ETIM', '$VOL', '$OP', '$FIL', '$SMNO', 'SPILL', '$SPILLOVER']

output = {}
for k in interesting_keys:
    if k in meta:
        if isinstance(meta[k], str):
            output[k] = meta[k]
        else:
            output[k] = "Complex Data (e.g. Matrix or Header)"

print("Total Metadata Keys:", len(meta.keys()))
print("Sample Metadata:")
print(json.dumps(output, indent=2))

# print first 20 keys just to see what's there
print("First 20 keys:", list(meta.keys())[:20])
