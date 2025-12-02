using Documenter, IterativeRefinement, LinearAlgebra

makedocs(
    modules = [IterativeRefinement],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == true),
    sitename = "IterativeRefinement.jl",
    pages = ["Overview" => "index.md",
             "Library" => "library.md"
             ]
)

# or maybe just the pkg site?
deploydocs(
    repo = "github.com/RalphAS/IterativeRefinement.jl.git",
)
