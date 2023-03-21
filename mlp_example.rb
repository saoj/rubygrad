require_relative 'lib/nn.rb'

nn = MLP.new(3, 4, 4, 1)

=begin

params = [
	[ -0.06137185896363029, -0.03307838956517317, 0.966756675458919, -0.5572641766005271 ],
	[ 0.8291385027448885, 0.8154728702930307, -0.30894738042121506, 0.6559606240987053 ],
	[ 0.22542349434058462, -0.5041560508924126, -0.5361203572824058, 0.09623335706358804 ],
	[ 0.36812656905262653, 0.07331959273406863, -0.1335322483054655, -0.4535584055875237 ]
],
[
	[ 0.005731625849950683, 0.21203497381769632, 0.3389414365363157, 0.19942286218653393, -0.7324914168170797 ],
	[ -0.699199117332054, -0.4147382334042651, 0.6819126698908244, 0.7196206732805, -0.6302488774956054 ],
	[ 0.634196466633496, -0.7643151860068604, -0.03620919399018563, 0.7611528933926044, 0.7484695299705786 ],
	[ -0.24942966770785713, 0.28844634480184395, 0.5601683653053287, 0.809130127543124, 0.7707590190416718 ]
],
[
	[ -0.10298569995244944, -0.12486701855289617, 0.42634703438074184, -0.2724602205795388, 0.6822836623046546 ]
]

nn.set_params(params)

=end

nn.show_params
puts

x_inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
y_expected = [1.0, -1.0, -1.0, 1.0] # desired

passes = 200
passes_format = "%#{passes.digits.length}d"
loss_precision = 10
loss_format = "%.#{loss_precision}f"
learning_rate = 0.2

(0...passes).each do |pass| 

    # forward pass
    y_calculated = x_inputs.map { |x| nn.calc(x, :tanh) }

    # loss function
    loss = 0.0
    y_expected.each_index { |i| loss += (y_calculated[i] - y_expected[i]) ** 2 }

    # backward pass
    nn.zero_grad
    loss.backward

    # improve
    nn.parameters.each { |p| p.value -= learning_rate * p.grad }

    puts "Pass #{passes_format % (pass + 1)} => Learning rate: #{"%.10f" % learning_rate} => Loss: #{loss_format % loss.value}" if (pass + 1) % 20 == 0 or pass == 0

    break if loss.value == 0 # just for fun just in case
end

y_calculated = x_inputs.map { |x| nn.calc(x, :tanh) }
puts
puts y_calculated
