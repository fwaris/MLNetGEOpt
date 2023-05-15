#r "nuget: Microsoft.ML.AutoML"
open Microsoft.ML.SearchSpace
let ss = SearchSpace()
ss.Add("a",Option.UniformDoubleOption(1.0,3.0))
