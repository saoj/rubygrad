# rubygrad
A port of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) to Ruby

## Objectives
- Learn neural networks, perceptrons, gradient descent, activation functions, backpropagation, etc.
- Learn Ruby and compare it with Python
- Evolve the library to add more features not necessarily present in [micrograd](https://github.com/karpathy/micrograd) for example the ability to specify a different activation function for each layer
- (_Work in Progress_) Use it to build a perceptron to recognize handwritten digits (see my other project [mnist-ready](https://github.com/saoj/mnist-ready))

## Installation

```ruby
gem install rubygrad
```

## Usage

```ruby
require 'rubygrad'
```

#### Binary Classification Example
```ruby
# Build a Machine Learning Perceptron with 4 layers
# First  Layer (Layer 0) => Input Layer => 3 Neurons => 3 Inputs
# Second Layer (Layer 1) => Hidden Layer => 4 Neurons
# Third  Layer (Layer 2) => Hidden Layer => 4 Neurons
# Fourth Layer (Layer 3) => Output Layer => 1 Neuron => 1 Output
nn = MLP.new(3, 4, 4, 1, :tanh)

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
    y_calculated = x_inputs.map { |x| nn.calc(x) }

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

y_calculated = x_inputs.map { |x| nn.calc(x) }
puts
puts "Final NN results:"
y_calculated.each_with_index { |y_c, i| puts "Output: #{y_c} => Expected: #{y_expected[i]}" }
```

#### Output
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

#### Specifying activation functions per each layer
```ruby
# All layers will use tanh
nn = MLP.new(3, 4, 4, 1, :tanh)
# or (equivalent)
nn = MLP.new(3, 4, 4, 1, [:tanh])

# Layer 0 => That's the input layer, so there is no activation function
# Layer 1 => sigmoid
# Layer 2 => sigmoid
# Layer 3 => tanh
nn = MLP.new(3, 4, 4, 1, [:sigmoid, :sigmoid, :tanh])
```

#### Showing weights and biases of the whole neural network
```ruby
nn = MLP.new(3, 4, 4, 1, [:sigmoid, :sigmoid, :tanh])

# show the neural net weights and biases as code
# you can easily save it for later and reload through the set_params method
nn.show_params

puts

# show the neural net weights and biases in words
nn.show_params(in_words = true)
```

#### Output
```
[
	[ 0.10716264032615364, -0.7071423208639602, 0.1163000965851324, 0.2701015638306199 ],
	[ 0.8748943671200455, -0.25715713613718627, -0.41685923836065575, -0.6043133638242268 ],
	[ 0.40159152554133537, -0.047327861996705245, 0.4876614339885963, 0.793143569774184 ],
	[ -0.28319370189054593, 0.6766873487030267, 0.8277741325778085, 0.3888260240294714 ]
],
[
	[ 0.8722769278933873, -0.9830828544066998, 0.715385878486424, -0.774099974211657, 0.7248963978186993 ],
	[ 0.3755762966825087, 0.03728211287511729, -0.04986344620919225, 0.3554754443267991, 0.7024771797583509 ],
	[ 0.46623333807978273, -0.053757185381019035, 0.03867611560991291, -0.11331576420042122, 0.31697978685612327 ],
	[ -0.23411955094766856, -0.23315628697626067, 0.11458612388308653, -0.20959685480548074, 0.372382068051728 ]
],
[
	[ 0.19761021225267417, -0.7834214164087676, 0.43077256716665757, -0.8111682816187338, -0.31730643380838086 ]
]

Layer 0: (3 inputs)
Layer 1: (4 neurons, :sigmoid activation)
	Neuron 1: (3 weights)
		Bias: 0.10716264032615364
		Weights: -0.7071423208639602, 0.1163000965851324, 0.2701015638306199
	Neuron 2: (3 weights)
		Bias: 0.8748943671200455
		Weights: -0.25715713613718627, -0.41685923836065575, -0.6043133638242268
	Neuron 3: (3 weights)
		Bias: 0.40159152554133537
		Weights: -0.047327861996705245, 0.4876614339885963, 0.793143569774184
	Neuron 4: (3 weights)
		Bias: -0.28319370189054593
		Weights: 0.6766873487030267, 0.8277741325778085, 0.3888260240294714
Layer 2: (4 neurons, :sigmoid activation)
	Neuron 1: (4 weights)
		Bias: 0.8722769278933873
		Weights: -0.9830828544066998, 0.715385878486424, -0.774099974211657, 0.7248963978186993
	Neuron 2: (4 weights)
		Bias: 0.3755762966825087
		Weights: 0.03728211287511729, -0.04986344620919225, 0.3554754443267991, 0.7024771797583509
	Neuron 3: (4 weights)
		Bias: 0.46623333807978273
		Weights: -0.053757185381019035, 0.03867611560991291, -0.11331576420042122, 0.31697978685612327
	Neuron 4: (4 weights)
		Bias: -0.23411955094766856
		Weights: -0.23315628697626067, 0.11458612388308653, -0.20959685480548074, 0.372382068051728
Layer 3: (1 neuron, :tanh activation)
	Neuron 1: (4 weights)
		Bias: 0.19761021225267417
		Weights: -0.7834214164087676, 0.43077256716665757, -0.8111682816187338, -0.31730643380838086
```

#### Setting all weights and biases manually
```ruby
nn = MLP.new(3, 4, 4, 1, [:sigmoid, :sigmoid, :tanh])

puts "Random:"
nn.show_params

# Now set to whatever you want:
all = [
	[ 0.10716264032615364, -0.7071423208639602, 0.1163000965851324, 0.2701015638306199 ],
	[ 0.8748943671200455, -0.25715713613718627, -0.41685923836065575, -0.6043133638242268 ],
	[ 0.40159152554133537, -0.047327861996705245, 0.4876614339885963, 0.793143569774184 ],
	[ -0.28319370189054593, 0.6766873487030267, 0.8277741325778085, 0.3888260240294714 ]
],
[
	[ 0.8722769278933873, -0.9830828544066998, 0.715385878486424, -0.774099974211657, 0.7248963978186993 ],
	[ 0.3755762966825087, 0.03728211287511729, -0.04986344620919225, 0.3554754443267991, 0.7024771797583509 ],
	[ 0.46623333807978273, -0.053757185381019035, 0.03867611560991291, -0.11331576420042122, 0.31697978685612327 ],
	[ -0.23411955094766856, -0.23315628697626067, 0.11458612388308653, -0.20959685480548074, 0.372382068051728 ]
],
[
	[ 0.19761021225267417, -0.7834214164087676, 0.43077256716665757, -0.8111682816187338, -0.31730643380838086 ]
]

# Now change it
nn.set_params(all)

puts
puts "Manually changed to:"
nn.show_params
```

#### Output
```
Random:
[
	[ -0.13912293308540957, 0.30268799442308425, -0.048102743649764745, -0.962411703704696 ],
	[ -0.18594220779286608, 0.7714512137011857, -0.03133572981131927, 0.9173322198149367 ],
	[ -0.33333231737453084, 0.4417777450715037, -0.3164673982895738, -0.022523457918021128 ],
	[ -0.4833437064977759, 0.16276526923408197, -0.3352383125781533, -0.9459722548359815 ]
],
[
	[ 0.044017554817758375, 0.9404455938103717, 0.27848433588752086, 0.26209587268564327, -0.044640343587271536 ],
	[ 0.8732254287953087, -0.4878795211779561, -0.831005253289361, -0.9618004107162326, 0.8324107561903806 ],
	[ -0.9078279123217432, -0.4412340056261552, 0.2606014164539314, 0.9319403191251423, -0.06260506603018401 ],
	[ 0.2095400809859027, 0.6137215231647983, 0.6669886944164458, -0.3014712110858331, 0.4514830155708711 ]
],
[
	[ -0.5331404474464982, -0.8014351154541197, -0.3600371014778567, -0.8361159398334321, 0.08851349359521499 ]
]

Manually changed to:
[
	[ 0.10716264032615364, -0.7071423208639602, 0.1163000965851324, 0.2701015638306199 ],
	[ 0.8748943671200455, -0.25715713613718627, -0.41685923836065575, -0.6043133638242268 ],
	[ 0.40159152554133537, -0.047327861996705245, 0.4876614339885963, 0.793143569774184 ],
	[ -0.28319370189054593, 0.6766873487030267, 0.8277741325778085, 0.3888260240294714 ]
],
[
	[ 0.8722769278933873, -0.9830828544066998, 0.715385878486424, -0.774099974211657, 0.7248963978186993 ],
	[ 0.3755762966825087, 0.03728211287511729, -0.04986344620919225, 0.3554754443267991, 0.7024771797583509 ],
	[ 0.46623333807978273, -0.053757185381019035, 0.03867611560991291, -0.11331576420042122, 0.31697978685612327 ],
	[ -0.23411955094766856, -0.23315628697626067, 0.11458612388308653, -0.20959685480548074, 0.372382068051728 ]
],
[
	[ 0.19761021225267417, -0.7834214164087676, 0.43077256716665757, -0.8111682816187338, -0.31730643380838086 ]
]
```
