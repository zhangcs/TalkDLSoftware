import mxnet as mx

# imperative style
print "Imperative Style Results:"
a = mx.nd.ones((2, 3))
print a
print a.asnumpy()
b = a * 2 + 1
print b.asnumpy()

print

# declarative style
print "Declarative Style Results:"
a = mx.sym.Variable('a')
print a
# print a.asnumpy() # a is a symbol now, empty
b = a * 2 + 1
c = b.bind(ctx=mx.cpu(), args={'a' : mx.nd.ones([2,3])})
c.forward()
print c.outputs[0].asnumpy()
