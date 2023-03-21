require 'rubygrad'

nn = MLP.new(3, 4, 4, 1)

x_inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
y_expected = [1.0, -1.0, -1.0, 1.0] # desired

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
puts y_calculated
