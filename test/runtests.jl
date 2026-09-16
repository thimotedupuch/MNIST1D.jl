using Test
using MNIST1D
using Statistics

@testset "MNIST1D" begin
    a = MNIST1D.dataset(num_samples=100, seed=7)
    b = MNIST1D.dataset(num_samples=100, seed=7)
    @test size(a.x) == (80, 40)
    @test size(a.x_test) == (20, 40)
    @test a.y == b.y && a.x == b.x
    @test sort(unique(a.y)) == collect(0:9)
    @test trainset(a) == (a.x, a.y)
    @test a.train == (x=a.x, y=a.y)
    @test a.test == (x=a.x_test, y=a.y_test)
    train_x, train_y = a.train
    @test train_x === a.x && train_y === a.y
    @test isapprox(std(vcat(a.x, a.x_test); corrected=false), 1; atol=1e-12)
    @test size(MNIST1D.dataset(num_samples=100, final_seq_length=17).x, 2) == 17
    @test_throws ArgumentError MNIST1D.dataset(num_samples=9)
end
