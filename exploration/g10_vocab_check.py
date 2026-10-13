"""Diagnostic: is the oracle's per-class bias explained by PROMPT VOCABULARY?

vad-exp.md cdv_prompts.py fixes the label tables (reproduced from VadCLIP):

  XD prompt words : B1=fighting  B2=shooting  B4=riot  B5=abuse
                    B6='car accident'  G=explosion
  UCF class words : Abuse Arrest Arson Assault Burglary Explosion Fighting
                    RoadAccidents Robbery Shooting Shoplifting Stealing Vandalism

The source model (UCF) was trained with UCF words.  When it scores XD it is fed
XD's prompt table, so two XD words -- "riot" and "car accident" -- are words the
UCF-trained text head never saw.

The G9 oracle produced  b = [B1 -1.0, B2 -0.8, B4 +1.0, B5 +1.0, B6 +0.9, G +1.0].
If the mechanism is prompt-vocabulary mismatch, the out-of-vocabulary classes
should carry the negative biases.

This script prints the correspondence so it can be judged by eye; it asserts
nothing, because a 6-element vector is too small to carry statistical weight on
its own -- the point is to name a testable hypothesis, not to claim a result.
"""
XD = {"B1": "fighting", "B2": "shooting", "B4": "riot",
      "B5": "abuse", "B6": "car accident", "G": "explosion"}
UCF = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
       "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting",
       "Stealing", "Vandalism"]
UCF_SET = {w.lower() for w in UCF}
ORACLE_B = {"B1": -1.0, "B2": -0.8, "B4": +1.0,
            "B5": +1.0, "B6": +0.9, "G": +1.0}

print("XD prompt word      in UCF vocab?   oracle b")
print("-" * 58)
neg_oov = neg_inv = 0
for code, word in XD.items():
    in_vocab = word.lower() in UCF_SET
    b = ORACLE_B[code]
    print(f"  {code:3s} {word:13s}  {'YES' if in_vocab else 'NO ':3s}"
          f"          {b:+.2f}")
    if b < 0 and not in_vocab:
        neg_oov += 1
    if b < 0 and in_vocab:
        neg_inv += 1

print()
print(f"negative-bias classes that are OUT of the UCF vocabulary: {neg_oov}")
print(f"negative-bias classes that are IN the UCF vocabulary    : {neg_inv}")
print()
print("Reading: B2 'shooting' IS in the UCF vocabulary yet carries a negative")
print("bias, so vocabulary mismatch is not a complete explanation.  It is a")
print("hypothesis worth one cheap test, not a conclusion.")