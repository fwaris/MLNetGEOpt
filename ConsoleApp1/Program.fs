//#load "packages.fsx"
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

let seClass = ctx.Auto().BinaryClassification(labelColumnName="Class",featureColumnName="Features",useFastTree=true,useLgbm=true)

let seBase =    
    let fac (ctx:MLContext) p = 
        let grpCols = Seq.append colInfr.ColumnInformation.CategoricalColumnNames  colInfr.ColumnInformation.NumericColumnNames |> Seq.toArray
        ctx.Transforms.Categorical.OneHotEncoding("EJ") <!>ctx.Transforms.Concatenate("Features",grpCols)
    SweepableEstimator(fac,new SearchSpace())

let seNorm (mbcMin,mbcMax) = 
    let lF = "fixZero"
    let lM = "maximumBinCount"
    let fac (ctx:MLContext) (p:Parameter) = 
        let fixZero = p.[lF].AsType<bool>()
        let mbc = p.[lM].AsType<int>() 
        let mbc = if mbc = 1 then None else Some(mbc)
        ctx.Transforms.NormalizeBinning("Features",fixZero=fixZero,?maximumBinCount=mbc) |> asEstimator        
    let ss = Search.init() |> Search.withChoice(lF,[|true;false|]) |> Search.withUniformInt(lM,mbcMin,mbcMax)
    SweepableEstimator(fac,ss)

let seGlobalContrast (sclMin,sclMax) = 
    let lZm = "ensureZeroMean"
    let lSd = "ensureUnitStandardDeviation"
    let lScl = "scale"
    let fac (ctx:MLContext) (p:Parameter) = 
        let ezm = p.[lZm].AsType<bool>()
        let eusd = p.[lSd].AsType<bool>() 
        let scale = p.[lScl].AsType<float32>()
        let scale = if scale = 0.f then None else Some scale
        ctx.Transforms.NormalizeGlobalContrast("Features",ensureZeroMean=ezm,ensureUnitStandardDeviation=eusd,?scale=scale) |> asEstimator
    let ss = 
        Search.init() 
        |> Search.withChoice(lZm,[|true;false|]) 
        |> Search.withChoice(lSd,[|true;false|])
        |> Search.withUniformFloat32(lScl,sclMin,sclMax)
    SweepableEstimator(fac,ss)

let seNormLogMeanVar = 
    let lucf = "useCdf"
    let fac (ctx:MLContext) (p:Parameter) = 
        let useCdf = p.[lucf].AsType<bool>()
        ctx.Transforms.NormalizeLogMeanVariance("Features",useCdf=useCdf) |> asEstimator
    let ss = Search.init() |> Search.withChoice(lucf,[|true;false|]) 
    SweepableEstimator(fac,ss)

let seNormLpNorm = 
    let lEzm = "ensureZeroMean"
    let lNorm = "norm"
    let fac (ctx:MLContext) (p:Parameter) = 
        let ezm = p.[lEzm].AsType<bool>()
        let norm = p.[lNorm].AsType<Microsoft.ML.Transforms.LpNormNormalizingEstimatorBase.NormFunction>()
        ctx.Transforms.NormalizeLpNorm("Features",norm=norm,ensureZeroMean=ezm) |> asEstimator
    let normVals = 
        Enum.GetValues(typeof<Microsoft.ML.Transforms.LpNormNormalizingEstimatorBase.NormFunction>) 
        |> Seq.cast<Microsoft.ML.Transforms.LpNormNormalizingEstimatorBase.NormFunction>
        |> Seq.map box
        |> Seq.toArray
    let ss = 
        Search.init() 
        |> Search.withChoice(lEzm,[|true;false|]) 
        |> Search.withChoice(lNorm, normVals)
    SweepableEstimator(fac,ss)


let seWhiten = 
    let lkind = "WhiteningKind"
    let lrank = "rank"
    let fac (ctx:MLContext) (p:Parameter) =         
        let kind = p.[lkind].AsType<Microsoft.ML.Transforms.WhiteningKind>()
        let rank = p.[lrank].AsType<int>()
        let rank = if rank = 0 then None else Some rank
        ctx.Transforms.VectorWhiten("Features",kind=kind,?rank=rank) |> asEstimator
    let wvals = 
        Enum.GetValues(typeof<Transforms.WhiteningKind>) 
        |> Seq.cast<Transforms.WhiteningKind>
        |> Seq.map box
        |> Seq.toArray
    let ss = 
        Search.init() 
        |> Search.withChoice(lkind,wvals) 
        |> Search.withUniformInt(lrank,0,10)
    SweepableEstimator(fac,ss)

let seNormMeanVar = 
    let lucf = "useCdf"
    let lEzm = "ensureZeroMean"
    let fac (ctx:MLContext) (p:Parameter) = 
        let useCdf = p.[lucf].AsType<bool>()
        let ezm = p.[lEzm].AsType<bool>()
        ctx.Transforms.NormalizeMeanVariance("Features",useCdf=useCdf,fixZero=ezm) |> asEstimator
    let ss = 
        Search.init() 
        |> Search.withChoice(lucf,[|true;false|]) 
        |> Search.withChoice(lEzm,[|true;false|])
    SweepableEstimator(fac,ss)
    
let g = 
    [
        Estimator seBase                                                                //Nil
        Alt [                                                                           //4
            Alt ([(1,10); (11,20); (21,30); (31,100)] |> List.map(seNorm>>Estimator))   //6
            Estimator seNormLpNorm
            Estimator seNormLogMeanVar
            Estimator seNormMeanVar
            Alt([0.1f .. 0.5f .. 4.0f] |> List.pairwise |> List.map(fun (a,b) -> a, b - 0.001f)  |> List.map(seGlobalContrast>>Estimator))
        ]
        Opt (Estimator seWhiten)                                                        //1
        Pipeline seClass
    ]

let expFac  (p:SweepablePipeline) =
    ctx.Auto()
        .CreateExperiment()
        .SetBinaryClassificationMetric(BinaryClassificationMetric.F1Score,"Class")
        .SetDataset(dv,3)
        .SetTrainingTimeInSeconds(60u*20u)        
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

let oPl,oAcc,rslt = Optimize.run 15000 CA.OptimizationKind.Maximize  expFac g




