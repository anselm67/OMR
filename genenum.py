

def gen_notes():
    for o in range(3, -1, -1):
        for idx_l, l in enumerate(['C', 'D', 'E', 'F', 'G', 'A', 'B']):
            print(f"    {l * (4-o)} = ({o}, {1 + idx_l})")
        print("")

    for o in range(4, 8):
        for idx_l, l in enumerate(['c', 'd', 'e', 'f', 'g', 'a', 'b']):
            print(f"    {l * (o-3)} = ({o}, {1 + idx_l})")
        print("")


gen_notes()
