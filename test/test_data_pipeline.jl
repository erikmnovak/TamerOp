using Test
using SparseArrays

const PM = PosetModules.Advanced
const IR = PM.IndicatorResolutions

@testset "Data pipeline: JSON round-trips" begin
    mktemp() do path, io
        close(io)
        data = PosetModules.PointCloud([[0.0], [1.0]])
        PosetModules.save_dataset_json(path, data)
        data2 = PosetModules.load_dataset_json(path)
        @test length(data2.points) == 2
        @test data2.points[2][1] == 1.0
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.ImageNd([0.0 1.0; 2.0 3.0])
        PosetModules.save_dataset_json(path, data)
        data2 = PosetModules.load_dataset_json(path)
        @test size(data2.data) == (2, 2)
        @test data2.data[2, 2] == 3.0
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.GraphData(3, [(1, 2), (2, 3)]; coords=[[0.0], [1.0], [2.0]], weights=[1.0, 2.0])
        PosetModules.save_dataset_json(path, data)
        data2 = PosetModules.load_dataset_json(path)
        @test data2.n == 3
        @test length(data2.edges) == 2
        @test data2.weights[2] == 2.0
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.EmbeddedPlanarGraph2D([[0.0, 0.0], [1.0, 0.0]], [(1, 2)])
        PosetModules.save_dataset_json(path, data)
        data2 = PosetModules.load_dataset_json(path)
        @test length(data2.vertices) == 2
        @test data2.edges[1] == (1, 2)
    end

    mktemp() do path, io
        close(io)
        cells = [Int[1]]
        boundaries = SparseMatrixCSC{Int,Int}[]
        grades = [Float64[0.0]]
        data = PosetModules.GradedComplex(cells, boundaries, grades)
        PosetModules.save_dataset_json(path, data)
        data2 = PosetModules.load_dataset_json(path)
        @test length(data2.cells_by_dim) == 1
        @test length(data2.grades) == 1
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.PointCloud([[0.0], [1.0]])
        spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
        PosetModules.save_pipeline_json(path, data, spec; degree=1)
        data2, spec2, degree2, _ = PosetModules.load_pipeline_json(path)
        @test length(data2.points) == 2
        @test spec2.kind == :rips
        @test spec2.params[:max_dim] == 1
        @test degree2 == 1
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.GraphData(3, [(1, 2), (2, 3)])
        spec = PosetModules.FiltrationSpec(kind=:graph_lower_star, vertex_grades=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        PosetModules.save_pipeline_json(path, data, spec; degree=0)
        _, spec2, degree2, _ = PosetModules.load_pipeline_json(path)
        @test spec2.kind == :graph_lower_star
        @test length(spec2.params[:vertex_grades]) == 3
        @test degree2 == 0
    end

    mktemp() do path, io
        close(io)
        data = PosetModules.PointCloud([[0.0], [1.0]])
        spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
        P, H, pi = PosetModules.ingest(data, spec; degree=0)
        PosetModules.save_encoding_json(path, P, H, pi)
        H2, pi2 = PosetModules.load_encoding_json(path; return_pi=true)
        @test PosetModules.nvertices(H2.P) == PosetModules.nvertices(P)
        @test PosetModules.axes_from_encoding(pi2) == PosetModules.axes_from_encoding(pi)
    end

    mktemp() do path, io
        close(io)
        F = PosetModules.face(1, [1])
        flats = [PosetModules.IndFlat(F, [0]; id=:F)]
        injectives = [PosetModules.IndInj(F, [0]; id=:E)]
        FG = PosetModules.Flange{PosetModules.QQ}(1, flats, injectives, [PosetModules.QQ(1)])
        enc = PosetModules.encode(FG; backend=:zn)
        PosetModules.save_encoding_json(path, enc.P, enc.H, enc.pi)
        H2, pi2 = PosetModules.load_encoding_json(path; return_pi=true)
        @test PosetModules.axes_from_encoding(pi2) == PosetModules.axes_from_encoding(enc.pi)
    end

    mktemp() do path, io
        close(io)
        Ups = [PosetModules.BoxUpset([0.0, 0.0])]
        Downs = [PosetModules.BoxDownset([1.0, 1.0])]
        enc = PosetModules.encode(Ups, Downs; backend=:pl_backend)
        PosetModules.save_encoding_json(path, enc.P, enc.H, enc.pi)
        H2, pi2 = PosetModules.load_encoding_json(path; return_pi=true)
        @test PosetModules.axes_from_encoding(pi2) == PosetModules.axes_from_encoding(enc.pi)
    end

    mktemp() do path, io
        close(io)
        sig_y = BitVector[
            BitVector([false, false]),
            BitVector([true,  false]),
            BitVector([true,  true]),
        ]
        sig_z = BitVector[
            BitVector([false]),
            BitVector([false]),
            BitVector([false]),
        ]
        P = PosetModules.ZnEncoding.SignaturePoset(sig_y, sig_z)
        U = PosetModules.upset_closure(P, trues(PosetModules.nvertices(P)))
        D = PosetModules.downset_closure(P, trues(PosetModules.nvertices(P)))
        H = PosetModules.FringeModule{PosetModules.QQ}(P, [U], [D], reshape([PosetModules.QQ(1)], 1, 1))
        PosetModules.save_encoding_json(path, H; include_leq=false)
        H2 = PosetModules.load_encoding_json(path)
        @test H2.P isa PosetModules.ZnEncoding.SignaturePoset
        @test PosetModules.nvertices(H2.P) == PosetModules.nvertices(P)
        @test PosetModules.leq_matrix(H2.P) == PosetModules.leq_matrix(P)
    end
end

@testset "Data pipeline: poset_from_axes kind=:grid" begin
    axes = (Float64[0.0, 1.0], Float64[0.0, 2.0, 4.0])
    P = PosetModules.poset_from_axes(axes; kind=:grid)
    @test P isa PosetModules.GridPoset
    @test PosetModules.nvertices(P) == 2 * 3
    @test PosetModules.leq(P, 1, 6)
    @test !PosetModules.leq(P, 6, 1)
end

@testset "Data pipeline: poset_from_axes kind=:grid with orientation -1" begin
    axes = (Float64[0.0, 1.0], Float64[0.0, 2.0, 4.0])
    P = PosetModules.poset_from_axes(axes; orientation=(1, -1), kind=:grid)
    @test P isa PosetModules.ProductOfChainsPoset
    @test PosetModules.nvertices(P) == 2 * 3
end

@testset "Interop adapters: GUDHI/Ripserer/Eirene" begin
    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "{\"simplices\": [[0],[1],[0,1]], \"filtration\": [0.0,0.0,1.0]}")
        end
        G = PosetModules.load_gudhi_json(path)
        @test length(G.grades) == 3
        @test length(G.boundaries) == 1
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "{\"simplices\": [[0],[1],[0,1]], \"filtration\": [0.0,0.0,1.0]}")
        end
        G = PosetModules.load_ripserer_json(path)
        @test length(G.grades) == 3
        @test length(G.boundaries) == 1
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "{\"simplices\": [[0],[1],[0,1]], \"filtration\": [0.0,0.0,1.0]}")
        end
        G = PosetModules.load_eirene_json(path)
        @test length(G.grades) == 3
        @test length(G.boundaries) == 1
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "# dim v1 v2 filtration\n")
            write(f, "0 0 0.0\n")
            write(f, "0 1 0.0\n")
            write(f, "1 0 1 1.0\n")
        end
        G = PosetModules.load_gudhi_txt(path)
        @test length(G.grades) == 3
        @test length(G.boundaries) == 1
    end
end

@testset "Interop adapters: Ripser/DIPHA matrix formats" begin
    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "0.0 1.0\n1.0 0.0\n")
        end
        G = PosetModules.load_ripser_distance(path; max_dim=1)
        @test length(G.grades) == 3
        @test maximum(getindex.(G.grades, 1)) == 1.0
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "0.0\n1.0 0.0\n")
        end
        G = PosetModules.load_ripser_lower_distance(path; max_dim=1)
        @test length(G.grades) == 3
        @test maximum(getindex.(G.grades, 1)) == 1.0
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "0.0 1.0\n0.0\n")
        end
        G = PosetModules.load_ripser_upper_distance(path; max_dim=1)
        @test length(G.grades) == 3
        @test maximum(getindex.(G.grades, 1)) == 1.0
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "0 1 1.0\n")
        end
        G = PosetModules.load_ripser_sparse_triplet(path; max_dim=1)
        @test length(G.grades) == 3
        @test maximum(getindex.(G.grades, 1)) == 1.0
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "0.0 0.0\n1.0 0.0\n")
        end
        pc = PosetModules.load_ripser_point_cloud(path)
        @test length(pc.points) == 2
        @test length(pc.points[1]) == 2
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, Float64[0.0, 1.0, 0.0])
        end
        G = PosetModules.load_ripser_binary_lower_distance(path; max_dim=1)
        @test length(G.grades) == 3
        @test maximum(getindex.(G.grades, 1)) == 1.0
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, Int64(8067171840))
            write(f, Int64(7))
            write(f, Int64(2))
            write(f, Float64[0.0, 1.0, 1.0, 0.0])
        end
        G = PosetModules.load_dipha_distance_matrix(path; max_dim=1)
        @test length(G.grades) == 3
        @test maximum(getindex.(G.grades, 1)) == 1.0
    end
end

@testset "Interop adapters: fixture round-trips" begin
    fixtures = joinpath(@__DIR__, "fixtures", "interop")

    G = PosetModules.load_gudhi_json(joinpath(fixtures, "gudhi.json"))
    @test length(G.grades) == 7
    @test length(G.boundaries) == 2

    G = PosetModules.load_ripserer_json(joinpath(fixtures, "ripserer.json"))
    @test length(G.grades) == 7
    @test length(G.boundaries) == 2

    G = PosetModules.load_eirene_json(joinpath(fixtures, "eirene.json"))
    @test length(G.grades) == 3
    @test length(G.boundaries) == 1

    G = PosetModules.load_gudhi_txt(joinpath(fixtures, "gudhi.txt"))
    @test length(G.grades) == 3
    @test length(G.boundaries) == 1

    G = PosetModules.load_ripserer_txt(joinpath(fixtures, "ripserer.txt"))
    @test length(G.grades) == 3
    @test length(G.boundaries) == 1

    G = PosetModules.load_eirene_txt(joinpath(fixtures, "eirene.txt"))
    @test length(G.grades) == 3
    @test length(G.boundaries) == 1

    G = PosetModules.load_ripser_distance(joinpath(fixtures, "ripser_distance.txt"); max_dim=1)
    @test length(G.grades) == 6
    @test maximum(getindex.(G.grades, 1)) == 3.0

    G = PosetModules.load_ripser_lower_distance(joinpath(fixtures, "ripser_lower_distance.txt"); max_dim=1)
    @test length(G.grades) == 6
    @test maximum(getindex.(G.grades, 1)) == 3.0

    G = PosetModules.load_ripser_upper_distance(joinpath(fixtures, "ripser_upper_distance.txt"); max_dim=1)
    @test length(G.grades) == 6
    @test maximum(getindex.(G.grades, 1)) == 3.0

    G = PosetModules.load_ripser_sparse_triplet(joinpath(fixtures, "ripser_sparse_triplet.txt"); max_dim=1)
    @test length(G.grades) == 6
    @test maximum(getindex.(G.grades, 1)) == 3.0

    pc = PosetModules.load_ripser_point_cloud(joinpath(fixtures, "ripser_point_cloud.txt"))
    @test length(pc.points) == 3
    @test length(pc.points[1]) == 2

    G = PosetModules.load_ripser_binary_lower_distance(joinpath(fixtures, "ripser_binary_lower_distance.bin"); max_dim=1)
    @test length(G.grades) == 6
    @test maximum(getindex.(G.grades, 1)) == 3.0

    G = PosetModules.load_dipha_distance_matrix(joinpath(fixtures, "dipha_distance_matrix.bin"); max_dim=1)
    @test length(G.grades) == 6
    @test maximum(getindex.(G.grades, 1)) == 3.0
end

@testset "Interop adapters: boundary complex + PModule JSON" begin
    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, """{
  "counts_by_dim": [2,1],
  "boundaries": [
    {"m":2,"n":1,"I":[1,2],"J":[1,1],"V":[1,-1]}
  ],
  "grades": [[0.0],[0.0],[1.0]]
}""")
        end
        G = PosetModules.load_boundary_complex_json(path)
        @test length(G.grades) == 3
        @test length(G.boundaries) == 1
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, """{
  "poset": {"n": 2, "leq": [[true,true],[false,true]]},
  "dims": [1,1],
  "edges": [
    {"src": 1, "dst": 2, "mat": [[1]]}
  ]
}""")
        end
        M = PosetModules.load_pmodule_json(path)
        @test M.dims == [1, 1]
    end
end

@testset "Interop adapters: streaming lower-triangular distance" begin
    fixtures = joinpath(@__DIR__, "fixtures", "interop")
    G = PosetModules.load_ripser_lower_distance_streaming(
        joinpath(fixtures, "ripser_lower_distance.txt"); radius=2.5
    )
    @test length(G.grades) == 5
    @test maximum(getindex.(G.grades, 1)) == 2.0
end

@testset "Data pipeline: JSON negative cases" begin
    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "{\"points\": [[0.0]]}")
        end
        @test_throws Exception PosetModules.load_dataset_json(path)
    end

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "{\"dataset\": {\"kind\": \"PointCloud\", \"points\": [[0.0]]}, \"spec\": {}}")
        end
        @test_throws Exception PosetModules.load_pipeline_json(path)
    end

    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([1.0, 0.0],))
    @test_throws Exception PosetModules.encode_from_data(data, spec; degree=0)

    cells = [Int[1]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.0], Float64[1.0]]
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    spec = PosetModules.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    @test_throws Exception PosetModules.encode_from_data(G, spec; degree=0)

    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.5],))
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    @test nvertices(enc.P) > 0

    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0, 1],), axis_kind=:rn)
    @test_throws Exception PosetModules.encode_from_data(data, spec; degree=0)

    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],), axis_kind=:zn)
    @test_throws Exception PosetModules.encode_from_data(data, spec; degree=0)
end

@testset "Data pipeline: quantization + coarsen axes" begin
    cells = [Int[1], Int[]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.05]]
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    spec = PosetModules.FiltrationSpec(kind=:graded, eps=0.1)
    enc = PosetModules.encode_from_data(G, spec; degree=0)
    @test PosetModules.axes_from_encoding(enc.pi)[1] == [0.0]

    data = PosetModules.PointCloud([[0.0], [1.0], [2.0], [3.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes_policy=:coarsen, max_axis_len=2)
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    @test length(PosetModules.axes_from_encoding(enc.pi)[1]) == 2
end

@testset "Data pipeline: cached poset reuse" begin
    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    enc1 = PosetModules.encode_from_data(data, spec; degree=0)
    enc2 = PosetModules.encode_from_data(data, spec; degree=0)
    @test enc1.P === enc2.P
end

@testset "Data pipeline: point cloud guardrails + witness" begin
    data = PosetModules.PointCloud([[0.0], [1.0], [2.0], [3.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=2, max_simplices=5)
    @test_throws Exception PosetModules.encode_from_data(data, spec; degree=0)

    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, sparse_rips=true, radius=1.1)
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    @test PosetModules.nvertices(enc.P) > 0

    spec = PosetModules.FiltrationSpec(kind=:witness, max_dim=1, landmarks=[1, 3])
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    @test PosetModules.nvertices(enc.P) > 0

    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, approx_rips=true, radius=1.1, max_edges=2, max_degree=1, sample_frac=1.0)
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    @test PosetModules.nvertices(enc.P) > 0
end

@testset "Data pipeline: flange emission (Zn)" begin
    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0, 1],), axis_kind=:zn)
    enc = PosetModules.encode_from_data(data, spec; degree=0, emit=:flange)
    @test haskey(enc.meta, :flange)
    FG = enc.meta[:flange]
    @test FG.n == 1

    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],), axis_kind=:rn)
    @test_throws Exception PosetModules.encode_from_data(data, spec; degree=0, emit=:flange)

    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    @test_throws Exception PosetModules.encode_from_data(data, spec; degree=0, emit=:flange)
end

@testset "Data pipeline: graded complex escape hatch" begin
    # single vertex
    cells = [Int[1]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.0]]
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    spec = PosetModules.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    enc = PosetModules.encode_from_data(G, spec; degree=0)
    @test enc.M.dims == [1, 1]

    # single edge between two vertices
    cells = [Int[1, 2], Int[1]]
    B1 = sparse([1, 2], [1, 1], [1, -1], 2, 1)
    boundaries = [B1]
    grades = [Float64[0.0], Float64[0.0], Float64[1.0]]
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    enc = PosetModules.encode_from_data(G, spec; degree=0)
    @test enc.M.dims == [2, 1]
    ri = PosetModules.rank_invariant(enc.M, PosetModules.InvariantOptions(); store_zeros=true)
    @test ri[(1, 2)] == 1

    # filled triangle: H1 appears at 1 and dies at 2
    cells = [Int[1, 2, 3], Int[1, 2, 3], Int[1]]
    B1 = sparse([1, 2, 1, 3, 2, 3],
                [1, 1, 2, 2, 3, 3],
                [-1, 1, -1, 1, -1, 1], 3, 3)
    B2 = sparse([1, 2, 3], [1, 1, 1], [1, -1, 1], 3, 1)
    boundaries = [B1, B2]
    grades = [Float64[0.0], Float64[0.0], Float64[0.0],
              Float64[1.0], Float64[1.0], Float64[1.0],
              Float64[2.0]]
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    spec2 = PosetModules.FiltrationSpec(kind=:graded, axes=([0.0, 1.0, 2.0],))
    enc1 = PosetModules.encode_from_data(G, spec2; degree=1)
    @test enc1.M.dims == [0, 1, 0]
end

@testset "Data pipeline: graded complex" begin
    cells = [Int[1]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.0]]
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    spec = PosetModules.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    enc = PosetModules.encode_from_data(G, spec; degree=0)
    @test enc.M.dims == [1, 1]

    H = PosetModules.fringe_presentation(enc.M)
    Mp = IR.pmodule_from_fringe(H)
    @test Mp.dims == enc.M.dims
end

@testset "Data pipeline: point cloud rips" begin
    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    @test enc.M.dims == [2, 1]
    bc = PosetModules.slice_barcode(enc.M, [1, 2])
    @test bc[(1, 2)] == 1
    @test bc[(1, 3)] == 1
end

@testset "Data pipeline: point cloud rips higher-dim" begin
    data = PosetModules.PointCloud([[0.0], [1.0], [2.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=2, axes=([0.0, 1.0, 2.0],))
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    @test enc.M.dims == [3, 1, 1]
end

@testset "Data pipeline: image lower-star" begin
    img = [0.0 1.0; 2.0 3.0]
    data = PosetModules.ImageNd(img)
    spec = PosetModules.FiltrationSpec(kind=:lower_star, axes=([0.0, 1.0, 2.0, 3.0],))
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    @test enc.M.dims == fill(1, 4)
end

@testset "Data pipeline: 3D cubical lower-star" begin
    img = reshape(Float64.(1:8), (2, 2, 2))
    data = PosetModules.ImageNd(img)
    spec = PosetModules.FiltrationSpec(kind=:lower_star, axes=([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],))
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    @test enc.M.dims == fill(1, 8)
end

@testset "Data pipeline: embedded planar graph toy" begin
    verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    edges = [(1, 2), (2, 3)]
    data = PosetModules.EmbeddedPlanarGraph2D(verts, edges)
    vgrades = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    spec = PosetModules.FiltrationSpec(kind=:graph_lower_star, vertex_grades=vgrades)
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    @test nvertices(enc.P) > 0
    @test PosetModules.locate(enc.pi, [0.0, 0.0]) > 0
    @test enc.H !== nothing
end

@testset "Data pipeline: wing distance bifiltration" begin
    verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    edges = [(1, 2), (2, 3)]
    data = PosetModules.EmbeddedPlanarGraph2D(verts, edges)
    spec = PosetModules.FiltrationSpec(
        kind=:wing_vein_bifiltration,
        grid=(8, 8),
        bbox=(0.0, 1.0, 0.0, 1.0),
        orientation=(-1, 1),
    )
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    @test enc.H !== nothing
    @test PosetModules.locate(enc.pi, [0.0, 0.0]) > 0
    @test PosetModules.locate(enc.pi, [-1.0, 0.0]) > 0
    @test PosetModules.locate(enc.pi, [-0.5, 0.5]) > 0
    ri = PosetModules.rank_invariant(enc.M, PosetModules.InvariantOptions(); store_zeros=true)
    @test ri[(1, 1)] >= 0

    opts = PosetModules.InvariantOptions(axes_policy=:encoding, strict=false, box=:auto)
    chain, _ = PosetModules.slice_chain(enc.pi, [-1.0, 0.0], [1.0, 1.0], opts; nsteps=25, check_chain=true)
    @test length(chain) > 0
    ri_chain = PosetModules.rank_invariant(enc.M, PosetModules.InvariantOptions(); store_zeros=true)
    for a in 1:length(chain)
        for b in a:length(chain)
            qa = chain[a]
            qb = chain[b]
            @test get(ri_chain, (qa, qb), 0) >= 0
        end
    end
end

@testset "Data pipeline: invariants compatibility" begin
    # Graded complex
    cells = [Int[1]]
    boundaries = SparseMatrixCSC{Int,Int}[]
    grades = [Float64[0.0]]
    G = PosetModules.GradedComplex(cells, boundaries, grades)
    spec = PosetModules.FiltrationSpec(kind=:graded, axes=([0.0, 1.0],))
    enc = PosetModules.encode_from_data(G, spec; degree=0)
    @test PosetModules.euler_surface(enc.M, enc.pi; axes_policy=:encoding) isa AbstractArray
    @test PosetModules.rank_invariant(enc.M, PosetModules.InvariantOptions()) isa Dict

    # Point cloud
    data = PosetModules.PointCloud([[0.0], [1.0]])
    spec = PosetModules.FiltrationSpec(kind=:rips, max_dim=1, axes=([0.0, 1.0],))
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    @test PosetModules.euler_surface(enc.M, enc.pi; axes_policy=:encoding) isa AbstractArray
    @test PosetModules.rank_invariant(enc.M, PosetModules.InvariantOptions()) isa Dict

    # Image (2D)
    img = [0.0 1.0; 2.0 3.0]
    data = PosetModules.ImageNd(img)
    spec = PosetModules.FiltrationSpec(kind=:lower_star, axes=([0.0, 1.0, 2.0, 3.0],))
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    @test PosetModules.euler_surface(enc.M, enc.pi; axes_policy=:encoding) isa AbstractArray

    # Graph (2D) for slice_chain
    data = PosetModules.GraphData(3, [(1, 2), (2, 3)])
    vgrades = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    spec = PosetModules.FiltrationSpec(kind=:graph_lower_star, vertex_grades=vgrades)
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    opts = PosetModules.InvariantOptions(axes_policy=:encoding, strict=false, box=:auto)
    chain, tvals = PosetModules.slice_chain(enc.pi, [0.0, 0.0], [1.0, 1.0], opts; nsteps=5)
    @test length(chain) > 0
    @test length(chain) == length(tvals)

    # Embedded planar graph (2D) for slice_chain
    verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    edges = [(1, 2), (2, 3)]
    data = PosetModules.EmbeddedPlanarGraph2D(verts, edges)
    vgrades = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    spec = PosetModules.FiltrationSpec(kind=:graph_lower_star, vertex_grades=vgrades)
    enc = PosetModules.encode_from_data(data, spec; degree=0)
    chain, tvals = PosetModules.slice_chain(enc.pi, [0.0, 0.0], [1.0, 1.0], opts; nsteps=5)
    @test length(chain) > 0
    @test length(chain) == length(tvals)
end
