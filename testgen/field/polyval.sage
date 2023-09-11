GF2.<x> = GF(2)[]
GF2_128.<y> = GF(2^128, modulus=x^128 + x^127 + x^126 + x^121 + 1, repr='int')

def int_repr(elem):
    val = 0
    for (i, coeff) in enumerate(elem.list()):
        val = val | int(coeff) << i
    return "{:#034x}".format(val)

a = GF2_128.random_element()
b = GF2_128.random_element()
c = a * b

print("1 =", int_repr(GF2_128(1)))
print("2^128 =", int_repr(y^128))
print("2^256 =", int_repr(y^256))
print("a =", int_repr(a))
print("b =", int_repr(b))
print("a * b =", int_repr(a * b))
print("a * a =", int_repr(a * a))
