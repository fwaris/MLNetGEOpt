﻿#load "packages.fsx"
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

let seBinClassification = ctx.Auto().BinaryClassification(labelColumnName="Class",featureColumnName="Features",useFastTree=true,useLgbm=true)

let seBase =    
    let fac (ctx:MLContext) p = 
        let grpCols = Seq.append colInfr.ColumnInformation.CategoricalColumnNames  colInfr.ColumnInformation.NumericColumnNames |> Seq.toArray
        ctx.Transforms.Categorical.OneHotEncoding("EJ") <!>ctx.Transforms.Concatenate("Features",grpCols)
    SweepableEstimator(fac,new SearchSpace())

let g = 
    [
        Estimator seBase       
        Opt(
            Alt [
                Estimator (E.seFtrSelCount 10)
                Estimator (E.seFtrSelMutualInf "Class")
            ])
        Opt(
            Alt [
                Alt ([(1,10); (11,20); (21,30); (31,100)] |> List.map(E.seNorm>>Estimator))
                Estimator E.seNormLpNorm
                Estimator E.seNormLogMeanVar
                Estimator E.seNormMeanVar
                Alt([0.1f .. 0.5f .. 4.0f] |> List.pairwise |> List.map(fun (a,b) -> a, b - 0.001f)  |> List.map(E.seGlobalContrast>>Estimator))
                Estimator E.seNormMinMax
                Estimator E.seNormRobustScaling
                Estimator (E.seNormSupBin "Class")
            ])
        Opt (Estimator E.seWhiten)
        Opt (
            Alt [
                Estimator (E.seProjPca (2,10))
                Estimator (E.seKernelMap (2,10))
            ])
        Pipeline seBinClassification
    ]

let expFac timeout (p:SweepablePipeline) =
    ctx.Auto()
        .CreateExperiment()
        .SetBinaryClassificationMetric(BinaryClassificationMetric.AreaUnderPrecisionRecallCurve,"Class")
        .SetDataset(dv,3)
        .SetTrainingTimeInSeconds(timeout)        
        .SetPipeline(p)
        .SetMonitor(
            let s : TrialSettings ref = ref Unchecked.defaultof<_>
            let printLine (isDone:bool) (r:TrialResult) =  printfn $"""M: {r.Metric} {isDone} - {s.Value.Parameter.["_pipeline_"]}"""
            {new IMonitor with
                 member this.ReportBestTrial(result) = printLine true result 
                 member this.ReportCompletedTrial(result) = printLine false result 
                 member this.ReportFailTrial(settings, ``exception``) = printfn "%A" ``exception``
                 member this.ReportRunningTrial(setting) = s.Value <- setting            
            })

//let p1 = Grammar.esimateGenomeSize g |> List.toArray |> Grammar.translate g |> (fst>>Grammar.toPipeline)
//p1.Estimators |> Seq.map(fun x->x.Value.SearchSpace) |> Seq.toArray
//let e1 = expFac p1
//e1.Run()

let oPl,oAcc = Optimize.run CA.OptimizationKind.Maximize (expFac 600u) g
Grammar.printPipeline ctx oPl
(*
let opLS = Grammar.toPipeline oPl
let trial = expFac 600u opLS
trial.Run()
*)

