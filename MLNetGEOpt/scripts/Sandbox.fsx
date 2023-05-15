#load "packages.fsx"
open Microsoft.ML
open Microsoft.ML.AutoML
open Microsoft.ML.SearchSpace
open MLUtils
open MLUtils.Pipeline
open MLNetGEOpt

let ctx = MLContext()

type T1 = {Features:float32[]}

let n1 =
    let fac (ctx:MLContext) p = ctx.Transforms.NormalizeBinning("features") |> asEstimator
    SweepableEstimator(fac,new SearchSpace())

let n2 = 
    let fac (ctx:MLContext) p = ctx.Transforms.NormalizeGlobalContrast("features") |> asEstimator
    SweepableEstimator(fac,new SearchSpace())

let dv = ctx.Data.LoadFromEnumerable<T1>([], Schema.cleanSchema typeof<T1>)

let facF1 (dv:IDataView) (ci:ColumnInformation)=ctx.Auto().Featurizer(dv,ci)    


let f1 = facF1 dv (ColumnInformation()) 

let gram2 = [Pipeline f1; Alt [Estimator n1; Estimator n2]]


