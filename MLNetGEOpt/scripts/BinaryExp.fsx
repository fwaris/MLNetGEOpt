#load "packages.fsx"
open System
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.AutoML
open Microsoft.ML.SearchSpace
open MLNetGEOpt
open MLUtils
open MLUtils.Pipeline

let path = @"E:\s\icr-identify-age-related-conditions\train.csv"

let ctx = MLContext()
let colInfr = ctx.Auto().InferColumns(path,labelColumnName="Class",groupColumns=false)
ColInfo.showCols colInfr.ColumnInformation
let ldr = ctx.Data.CreateTextLoader(colInfr.TextLoaderOptions)
let dv = ldr.Load(path)

let settings = BinaryExperimentSettings()
settings.OptimizingMetric <- BinaryClassificationMetric.AreaUnderPrecisionRecallCurve
settings.Trainers.Clear()
settings.Trainers.Add(BinaryClassificationTrainer.FastTree)
settings.MaxExperimentTimeInSeconds <- 600u
let exp1 = ctx.Auto().CreateBinaryClassificationExperiment(settings)
let pgr =
    {new IProgress<CrossValidationRunDetail<BinaryClassificationMetrics>> with
         member this.Report(value) =
            let avg = value.Results |> Seq.averageBy(fun x->x.ValidationMetrics.AreaUnderPrecisionRecallCurve)
            printfn $"AUPRC: {avg}"
    }
let rslt = exp1.Execute(dv,4u,columnInformation=colInfr.ColumnInformation,progressHandler=pgr)
rslt.BestRun.Results |> Seq.averageBy(fun x ->x.ValidationMetrics.AreaUnderPrecisionRecallCurve)
let m1 = rslt.BestRun.Results |> Seq.head |> fun h -> h.Model
MLUtils.Schema.printTxChain dv.Schema 2 m1
let dvs = m1.Transform(dv)
type [<CLIMutable>] T1 = {Class:bool; Probabiliy:float32; PredictedLabel:bool}
let t1s = ctx.Data.CreateEnumerable<T1>(dvs,false,ignoreMissingColumns=true) |> Seq.toArray
