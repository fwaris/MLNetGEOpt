namespace MLNetGEOpt
open System
open Microsoft.ML.AutoML
open Microsoft.ML.SearchSpace
open Microsoft.ML

type Search =
    static member init() = new SearchSpace()
    static member withUniformFloat(s:string,lo,hi,?logBase,?defaultValue)  = fun (ss:SearchSpace) -> ss.Add(s,Option.UniformDoubleOption(lo,hi,?logBase=logBase,?defaultValue=defaultValue)); ss
    static member withUniformInt (s:string,lo,hi,?logBase,?defaultValue) =  fun (ss:SearchSpace) -> ss.Add(s,Option.UniformIntOption(lo,hi,?logBase=logBase,?defaultValue=defaultValue)); ss
    static member withUniformFloat32 (s:string,lo,hi,?logBase,?defaultValue) = fun (ss:SearchSpace) -> ss.Add(s,Option.UniformSingleOption(lo,hi,?logBase=logBase,?defaultValue=defaultValue)); ss
    static member withChoice(s:string,choices: obj[],?defaultChoice) = fun  (ss:SearchSpace) -> let opt = match defaultChoice with Some d -> Option.ChoiceOption(choices,d) | _-> Option.ChoiceOption(choices) in ss.Add(s,opt); ss
