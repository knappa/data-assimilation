# Data assimilation using the Extended Kalman Filter (EKF)

As applied to the models in "COVID-19 virtual patient cohort suggests immune mechanisms driving disease outcomes" by _Jenner et al._

https://doi.org/10.1371/journal.ppat.1009753

https://github.com/adriannejnner/COVID19-Virtual-Trial-PLOS-Pathogens

# Usage

Currently in active development, so these descriptions may be unstable and change without warning.

Developed using [Julia 1.10.1](https://julialang.org/downloads/). You may find [juliaup](https://github.com/JuliaLang/juliaup) useful for keeping track of Julia versions.

Assuming the `julia` executable is in your path, there are three scripts intended for direct use.

```
julia personalize_patient_data.jl <patient index>
```

Parameter selects the patient number (first is 1) from the Jenner dataset. Learning is done using the full model. Does not currently work.


```
julia personalize_virt_patient.jl 
```

Generates a virtual patient under the full model and attempts to learn based on this trajectory. Produces a collection of graphs in pdf format and jld2 files containing state variable and parameter distributions. Does not currently work. (Will convert to non-log representation)

```
julia personalize_patient_virt_simple_model.jl 
```

Similar to above, but using the simplified model. KF done not on log-coords, but on the original state variables. (Log coords resulted in stability problems) Presents a theoretical problem: what to do about the boundary at 0 for state/params? Practical solution so far is to take absolute value. https://en.wikipedia.org/wiki/Folded_normal_distribution It works OK. This looks better: https://en.wikipedia.org/wiki/Modified_half-normal_distribution but there is more to explore here. Transforms via hinge functions? $\ln(1+\exp(x))$ These are approximately linear for $x \gg 0$

```
julia personalize_patient_virt_ifn.jl 
```

Same as above but for the version of the model with IFN dynamics. (Middle complexity)