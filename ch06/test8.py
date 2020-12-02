#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class MulLayer:

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y
        return out

    def backward(self, dout):
        dy = dout*self.x
        dx = dout*self.y
        return dx, dy


class AddLayer:

    def __init__(self):
        pass

    def forward(self, x, y):
        out = x+y
        return out

    def backward(self, dout):
        return dout, dout


'''
apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)
'''

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
mul_orange_layer = MulLayer()
add_layer = AddLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
print(price)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
print(dapple, dapple_num, dorange, dorange_num, dtax)
