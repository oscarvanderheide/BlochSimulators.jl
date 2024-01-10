using BlochSimulators
using Documenter
using Literate

# convert example files to markdown with Literate
Literate.markdown("docs/examples/dictionary.jl", "docs/src"; flavor=Literate.CommonMarkFlavor())
Literate.markdown("docs/examples/signal.jl", "docs/src"; flavor=Literate.CommonMarkFlavor())
Literate.markdown("docs/examples/adiabatic.jl", "docs/src"; flavor=Literate.CommonMarkFlavor())

DocMeta.setdocmeta!(BlochSimulators, :DocTestSetup, :(using BlochSimulators); recursive=true)

makedocs(;
    modules=[BlochSimulators],
    authors="Oscar van der Heide <o.vanderheide@umcutrecht.nl> and contributors",
    repo="https://github.com/oscarvanderheide/BlochSimulators.jl/blob/{commit}{path}#{line}",
    sitename="BlochSimulators.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages = [
        "Home" => "index.md",
        "Overview" => "overview.md",
        "Examples" => ["dictionary.md", "signal.md", "adiabatic.md"],
        "API" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/oscarvanderheide/BlochSimulators.jl.git",
)

