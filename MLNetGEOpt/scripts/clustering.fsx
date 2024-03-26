#load "packages.fsx"
open System.Diagnostics
open System
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.AutoML
open Microsoft.ML.SearchSpace
open MLNetGEOpt
open MLUtils
open MLUtils.Pipeline

let ctx = MLContext()
let path = @"C:\s\data.dv"

let dv = ctx.Data.LoadFromBinary path
let dvTrain,dvTest = let tt = ctx.Data.TrainTestSplit(dv,0.1) in tt.TrainSet, tt.TestSet
dv.Schema |> Seq.iter (fun x -> printfn $""" "{x.Name}" //{x.Type} """)

let numCols = 
    [
        "num1"
        "num2"
    ]

let catCols = 
    [
        "cat1"        
    ]

let ignoreCols = 
    [
        "ban" //String 
    ]

let colInfo = ColumnInformation()
ColInfo.setAsCategorical catCols colInfo
ColInfo.setAsNumeric numCols colInfo
ColInfo.ignore ignoreCols
colInfo.LabelColumnName <- "ban"

let featurizer() =  ctx.Auto().Featurizer(dv, colInfo, E.FEATURES)

let seConvertNum() = 
    let fac (ctx:MLContext) p =                 
        let numCols = numCols |> List.map InputOutputColumnPair |> List.toArray        
        ctx.Transforms.Conversion.ConvertType(numCols, outputKind=DataKind.Single) 
        |> asEstimator
    let ss = 
        Search.init() 
        |> Search.withId $"Convert to single"
    SweepableEstimator(fac,ss)

let seConcat() =    
    let fac (ctx:MLContext) p =         
        let allCols = numCols @ catCols |> List.toArray
        let catCols = catCols |> List.map InputOutputColumnPair |> List.toArray        
        let tx1H= ctx.Transforms.Categorical.OneHotEncoding(catCols)
        let txConcat = ctx.Transforms.Concatenate(E.FEATURES, allCols)
        tx1H <!> txConcat
    SweepableEstimator(fac,Search.init() |> Search.withId $"1H and concat")

//returns sweepable estimator for KMeans with k as a parameter
//create a new estimator for each k and add to the 'grammar'
let seCluster (k:int) ()=         
    let fac (ctx:MLContext) (p:Parameter) = 
        let opts = Trainers.KMeansTrainer.Options()
        opts.FeatureColumnName <- E.FEATURES
//        opts.MaximumNumberOfIterations <- 500
        opts.NumberOfClusters <- k    
        printfn $"k={k}"
        ctx.Clustering.Trainers.KMeans(opts) 
        |> asEstimator              
    let ss = 
        Search.init() 
        |> Search.withId $"Cluster size {k}"        
    SweepableEstimator(fac,ss)

//individual field normalizers
let fieldNorm col = 
        Alt [
            Alt ([None; yield! [for i in 10 .. 10 .. 100 -> Some i]] 
            |> List.collect(fun maxBins -> 
                [
                    Estimator(Eh.seNormBin col true maxBins)
                    Estimator(Eh.seNormBin col false maxBins)
                ]))
            Estimator (Eh.seNormLogMeanVar col true)
            Estimator (Eh.seNormLogMeanVar col false)
            Estimator (Eh.seNormMeanVar col true true)
            Estimator (Eh.seNormMeanVar col true false)
            Estimator (Eh.seNormMeanVar col false true)
            Estimator (Eh.seNormMeanVar col false false)
            Estimator (Eh.seNormMinMax col true)
            Estimator (Eh.seNormMinMax col false)
            Estimator (Eh.seNormRobustScaling col true)
            Estimator (Eh.seNormRobustScaling col false)
        ]
        
let g = 
    [
        Estimator seConvertNum
        yield! numCols |> List.map fieldNorm 
        Estimator seConcat        
        Opt(
            Alt [
                Estimator (Eh.Def.seGlobalContrast true true None)
                Estimator (Eh.Def.seGlobalContrast true false None)
                Estimator (Eh.Def.seGlobalContrast false true None)
                Estimator (Eh.Def.seGlobalContrast false false None)
                Estimator (Eh.Def.seNormLpNorm None true)
                Estimator (Eh.Def.seNormLpNorm None false)
            ]
        )
        Alt [for i in 3 .. 20 -> Estimator (seCluster i)]  // this works 
    ]

let runTrial (pipeline:SweepablePipeline) (settings:TrialSettings) =     
    task {
        let stopwatch = Stopwatch()
        let parameter = settings.Parameter[E.PIPELINE]
        // Use parameters to build pipeline
        let pipeline = pipeline.BuildFromOption(ctx, parameter)
        let model = pipeline.Fit(dvTrain)
        let scored = model.Transform(dvTest)
        let eval = 
            ctx.Clustering.Evaluate(
                data = scored,
                labelColumnName = "PredictedLabel",
                scoreColumnName = "Score",
                featureColumnName = E.FEATURES)
        return new TrialResult
                (
                    Metric = eval.DaviesBouldinIndex,
                    Model = model,
                    TrialSettings = settings,
                    DurationInMilliseconds = float stopwatch.ElapsedMilliseconds
                )

    }

type KRunner (pipeline:SweepablePipeline) =
    interface ITrialRunner with
        member this.Dispose() = ()
        member this.RunAsync(settings, ct) = runTrial pipeline settings

let expFac timeout (p:SweepablePipeline) =
    ctx.Auto()
        .CreateExperiment()
        .SetRegressionMetric(RegressionMetric.MeanAbsoluteError)        
        .SetDataset(dv,3)        
        .SetTrainingTimeInSeconds(timeout)        
        .SetPipeline(p)
        .SetTrialRunner(new KRunner(p))
        .SetMonitor(
            let s : TrialSettings ref = ref Unchecked.defaultof<_>
            let printLine (isDone:bool) (r:TrialResult) =  
                printfn $"""M: {r.Metric} {isDone} - {s.Value.Parameter.[E.PIPELINE]}"""
            {new IMonitor with
                 member this.ReportBestTrial(result) = printLine true result 
                 member this.ReportCompletedTrial(result) = printLine false result 
                 member this.ReportFailTrial(settings, ``exception``) = 
                    printfn "%A" ``exception``
                 member this.ReportRunningTrial(setting) = s.Value <- setting            
            })

let genomeSize = 50 // Grammar.estimateGenomeSize g + 1
let oPl,oAcc,rlst,cache = 
    Optimize.run 
        genomeSize 
        100000 
        CA.OptimizationKind.Minimize 
        (expFac (10u * 60u)) 
        g
;;
printfn $"Metric {oAcc}"
printfn $"Cache size {cache.Count}"
Grammar.printPipeline ctx oPl
rlst.TrialSettings
