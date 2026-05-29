import os
import fcsparser

def main():
    neg_fcs = r"data\Experiment 2026!05!27 9!30\24 Tube Rack (5mL) - 1\negative\B01 Well - B01.fcs"
    calcein_fcs = r"data\Experiment 2026!05!27 9!30\24 Tube Rack (5mL) - 1\Calcein\A01 Well - A01.fcs"

    if not os.path.exists(neg_fcs) or not os.path.exists(calcein_fcs):
        print("FCS files not found.")
        return

    print("=== Parsing Negative FCS Metadata ===")
    meta_neg = fcsparser.parse(neg_fcs, meta_data_only=True)
    
    print("=== Parsing Calcein FCS Metadata ===")
    meta_cal = fcsparser.parse(calcein_fcs, meta_data_only=True)

    print("\nAll Negative metadata keys:")
    for k in sorted(meta_neg.keys()):
        print(f"  {k}: {str(meta_neg[k])[:100]}")
        
    print("\nComparing different metadata values:")
    for k in sorted(meta_neg.keys()):
        if k in meta_cal:
            if meta_neg[k] != meta_cal[k]:
                print(f"  {k:<15} Neg: {str(meta_neg[k])[:40]:<40} Cal: {str(meta_cal[k])[:40]}")
        else:
            print(f"  {k:<15} Neg: {str(meta_neg[k])[:40]:<40} Cal: [Missing]")

if __name__ == "__main__":
    main()
