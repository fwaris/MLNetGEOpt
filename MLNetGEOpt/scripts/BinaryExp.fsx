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
let colInfr = ctx.Auto().InferColumns(path,labelColumnName="Class",groupColumns=true)
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
let epsilon = 10.e-15
let m1 = rslt.BestRun.Results |> Seq.head |> fun h -> h.Model
MLUtils.Schema.printTxChain dv.Schema 2 m1
type [<CLIMutable>] T1 = {Id:string; __Features__:float32[]; Class:bool; Probabiliy:float32; PredictedLabel:bool; Score:float32}
let dvs = m1.Transform(dv)
dvs.Schema |> Schema.printSchema

let t1s = ctx.Data.CreateEnumerable<T1>(dvs,false,ignoreMissingColumns=true) |> Seq.toArray
open Plotly.NET
t1s |> Seq.map(fun x->x.Score) |> Chart.Violin |> Chart.show
t1s |> Seq.map(fun x->x.Probabiliy) |> Chart.Violin |> Chart.show

let c0s = t1s |> Seq.filter(fun x->x.Class)
let c1s = t1s |> Seq.filter(fun x->not x.Class)

[
    c0s |> Seq.map(fun x->x.Score) |> Chart.Violin |> Chart.withTraceInfo $"True {Seq.length c0s}"
    c1s |> Seq.map(fun x->x.Score) |> Chart.Violin |> Chart.withTraceInfo $"False {Seq.length c1s}"
]
|> Chart.combine 
|> Chart.withXAxisStyle("Raw score")
|> Chart.withTitle("AUCPR: 0.836")
|> Chart.show

let n0 = c0s |> Seq.length |> float
let n1 = c1s |> Seq.length |> float
let prev0 = 0.5 // n0 / (n0 + n1) 
let prev1 = 0.5 //n1 / (n0 + n1)
let w0 = 1.0 / prev0
let w1 = 1.0 / prev1
let t0 = c0s |> Seq.sumBy(fun x -> log (1.0 - epsilon))
let t1 = c1s |> Seq.sumBy(fun x -> log epsilon)
let ll = (-(w0/n0 * t0) - (w1/n1 * t1))/ (w0 + w1)
log epsilon
log (1.0 - epsilon)

let clbtr = ctx.BinaryClassification.Calibrators.Naive(labelColumnName="Class",scoreColumnName="Score")
let dvsP = clbtr.Fit(dvs).Transform(dvs)
let t2s = ctx.Data.CreateEnumerable<T1>(dvsP,false,ignoreMissingColumns=true) |> Seq.toArray
[
    t2s |> Seq.map(fun x->x.Score) |> Chart.Violin 
    t2s |> Seq.map(fun x->x.Probabiliy) |> Chart.Violin
]
|> Chart.combine |> Chart.show

let g1 = ctx.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName="Class",featureColumnName="Score")
let g1m = g1.Fit(dvs)
