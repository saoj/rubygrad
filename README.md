# rubygrad
A port of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) to Ruby

## Objectives
- Learn neural networks, perceptrons, gradient descent, activation functions, backpropagation, etc.
- Learn Ruby and compare it with Python
- Use it to build a perceptron to recognize handwritten digits (see my other project [mnist-ready](https://github.com/saoj/mnist-ready))

## Installation

```ruby
gem install rubygrad
```

## Usage

```ruby
require 'rubygrad'
```

### Binary Classification Example
```ruby
# Build a Machine Learning Perceptron with 4 layers
# First  Layer (Layer 0) => Input Layer => 3 Neurons => 3 Inputs
# Second Layer (Layer 1) => Hidden Layer => 4 Neurons
# Third  Layer (Layer 2) => Hidden Layer => 4 Neurons
# Fourth Layer (Layer 3) => Output Layer => 1 Neuron => 1 Output
nn = MLP.new(3, 4, 4, 1)

# 4 input samples
x_inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

# expected output for each of the 4 inputs above
y_expected = [1.0, -1.0, -1.0, 1.0]

passes = 2000
learning_rate = 0.2

_loss_precision = 10
_passes_format = "%#{passes.digits.length}d"
_loss_format = "%.#{_loss_precision}f"

(0...passes).each do |pass| 

    # forward pass (calculate output)
    y_calculated = x_inputs.map { |x| nn.calc(x, :tanh) }

    # loss function (check how good the neural net is)
    loss = 0.0
    y_expected.each_index { |i| loss += (y_calculated[i] - y_expected[i]) ** 2 }

    # backward pass (calculate gradients)
    nn.zero_grad
    loss.backward

    # improve neural net (update weights and biases)
    nn.parameters.each { |p| p.value -= learning_rate * p.grad }

    puts "Pass #{_passes_format % (pass + 1)} => Learning rate: #{"%.10f" % learning_rate} => Loss: #{_loss_format % loss.value}" if (pass + 1) % 100 == 0 or pass == 0

    break if loss.value == 0 # just for fun and just in case
end

y_calculated = x_inputs.map { |x| nn.calc(x, :tanh) }
puts
puts "Final NN results:"
y_calculated.each_with_index { |y_c, i| puts "Output: #{y_c} => Expected: #{y_expected[i]}" }
```

### Output
```
Pass    1 => Learning rate: 0.2000000000 => Loss: 6.4414947529
Pass  100 => Learning rate: 0.2000000000 => Loss: 0.0014086315
Pass  200 => Learning rate: 0.2000000000 => Loss: 0.0007669303
Pass  300 => Learning rate: 0.2000000000 => Loss: 0.0005244376
Pass  400 => Learning rate: 0.2000000000 => Loss: 0.0003974497
Pass  500 => Learning rate: 0.2000000000 => Loss: 0.0003194754
Pass  600 => Learning rate: 0.2000000000 => Loss: 0.0002667991
Pass  700 => Learning rate: 0.2000000000 => Loss: 0.0002288631
Pass  800 => Learning rate: 0.2000000000 => Loss: 0.0002002596
Pass  900 => Learning rate: 0.2000000000 => Loss: 0.0001779336
Pass 1000 => Learning rate: 0.2000000000 => Loss: 0.0001600304
Pass 1100 => Learning rate: 0.2000000000 => Loss: 0.0001453591
Pass 1200 => Learning rate: 0.2000000000 => Loss: 0.0001331204
Pass 1300 => Learning rate: 0.2000000000 => Loss: 0.0001227580
Pass 1400 => Learning rate: 0.2000000000 => Loss: 0.0001138729
Pass 1500 => Learning rate: 0.2000000000 => Loss: 0.0001061715
Pass 1600 => Learning rate: 0.2000000000 => Loss: 0.0000994331
Pass 1700 => Learning rate: 0.2000000000 => Loss: 0.0000934883
Pass 1800 => Learning rate: 0.2000000000 => Loss: 0.0000882054
Pass 1900 => Learning rate: 0.2000000000 => Loss: 0.0000834802
Pass 2000 => Learning rate: 0.2000000000 => Loss: 0.0000792291

Final NN results:
Output: 0.9946071757394565 => Expected: 1.0
Output: -0.9970474246057773 => Expected: -1.0
Output: -0.9960119927936328 => Expected: -1.0
Output: 0.9949518028977203 => Expected: 1.0
```
