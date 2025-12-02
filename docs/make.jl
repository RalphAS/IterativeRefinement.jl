using Documenter, IterativeRefinement, LinearAlgebra

DocMeta.setdocmeta!(IterativeRefinement, :DocTestSetup, :(using IterativeRefinement);
                    recursive=true)

makedocs(
    modules = [IterativeRefinement],
    sitename = "IterativeRefinement.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://RalphAS.github.io/IterativeRefinement.jl",
        assets=String[],
    ),
    pages = ["Overview" => "index.md",
             "Library" => "library.md"
             ]
)

# or maybe just the pkg site?
deploydocs(
    repo = "github.com/RalphAS/IterativeRefinement.jl",
)
