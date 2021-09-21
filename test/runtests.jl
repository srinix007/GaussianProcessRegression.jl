using GaussianProcessRegression
using Test

kerns = (SquaredExp(),)

@testset "Kernel Implementation" for kern in kerns

    @testset "dim $dim" for dim in 1:5
        x = rand(dim, 100)
        xp = rand(dim, 200)
        hp = init_hp(kern)

    end

end

    