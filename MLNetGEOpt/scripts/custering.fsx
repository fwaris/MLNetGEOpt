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
        "num1" // not actual names
        "num2"
    ]

let catCols = 
    [
        "cat1" //Int32 
    ]

let ignoreCols = 
    [
        "id" //String 
    ]

let colInfo = ColumnInformation()
ColInfo.setAsCategorical catCols colInfo
ColInfo.setAsNumeric numCols colInfo
ColInfo.ignore ignoreCols
colInfo.LabelColumnName <- "ban"

let featurizer() =  ctx.Auto().Featurizer(dv, colInfo, E.FEATURES)

let seBase() =    
    let fac (ctx:MLContext) p =         
        let allCols = numCols @ catCols |> List.toArray
        let numCols = numCols |> List.map InputOutputColumnPair |> List.toArray
        let catCols = catCols |> List.map InputOutputColumnPair |> List.toArray        
        let txConv = ctx.Transforms.Conversion.ConvertType(numCols, outputKind=DataKind.Single)
        let tx1H= ctx.Transforms.Categorical.OneHotEncoding(catCols)
        let txConcat = ctx.Transforms.Concatenate(E.FEATURES, allCols)
        txConv <!> tx1H <!> txConcat
    SweepableEstimator(fac,Search.init() |> Search.withId $"Convert and concat")

//returns sweepable estimator for KMeans
//k is part of the search space
//NOTE: the search space is not explored fully - search hovers around default value
let seClusterWithSS () = 
    let K = "k"
    let fac (ctx:MLContext) (p:Parameter) = 
        let k = p.[K].AsType<int>()
        let opts = Trainers.KMeansTrainer.Options()
        opts.FeatureColumnName <- E.FEATURES
        opts.MaximumNumberOfIterations <- 500
        opts.NumberOfClusters <- k
        printfn $"k={k}"
        ctx.Clustering.Trainers.KMeans(opts) |> asEstimator              
    let ss = 
        Search.init() 
        |> Search.withId $"Cluster size"
        |> Search.withUniformInt(K, 3, 20, defaultValue=10)        
    SweepableEstimator(fac,ss)

//returns sweepable estimator for KMeans with k as a parameter
//create a new estimator for each k and add to the 'grammar'
let seCluster (k:int) ()=         
    let fac (ctx:MLContext) (p:Parameter) = 
        let opts = Trainers.KMeansTrainer.Options()
        opts.FeatureColumnName <- E.FEATURES
        opts.MaximumNumberOfIterations <- 500
        opts.NumberOfClusters <- k    
        printfn $"k={k}"
        ctx.Clustering.Trainers.KMeans(opts) |> asEstimator              
    let ss = 
        Search.init() 
        |> Search.withId $"Cluster size {k}"        
    SweepableEstimator(fac,ss)

let g = 
    [
        Estimator seBase
        //Pipeline featurizer
        Opt(Estimator (E.Def.seFtrSelCount 3))        
        Alt [
            Alt ([(1,10); (11,20); (21,30); (31,100)] |> List.map(E.Def.seNorm>>Estimator))
            Estimator E.Def.seNormLpNorm
            Estimator E.Def.seNormLogMeanVar
            Estimator E.Def.seNormMeanVar
            Alt([0.1f .. 0.5f .. 4.0f] |> List.pairwise |> List.map(fun (a,b) -> a, b - 0.001f)  |> List.map(E.Def.seGlobalContrast>>Estimator))
            Estimator E.Def.seNormMinMax
            Estimator E.Def.seNormRobustScaling
        ]
//        Opt (Estimator E.Def.seWhiten)
        // Opt (
        //    Alt [
        //        Estimator (E.Def.seProjPca (2,10))
        //        Estimator (E.Def.seKernelMap (2,10))
        //    ])
        Alt [for i in 3 .. 20 -> Estimator (seCluster i)]  // this works 
        //Estimator seClusterWithSS                        // this does not work
    ]

//run a trial
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

//trial runner wrapper class
type KRunner (pipeline:SweepablePipeline) =
    interface ITrialRunner with
        member this.Dispose() = ()
        member this.RunAsync(settings, ct) = runTrial pipeline settings

//experiment factory
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
            let printLine (isDone:bool) (r:TrialResult) =  printfn $"""M: {r.Metric} {isDone} - {s.Value.Parameter.[E.PIPELINE]}"""
            {new IMonitor with
                 member this.ReportBestTrial(result) = printLine true result 
                 member this.ReportCompletedTrial(result) = printLine false result 
                 member this.ReportFailTrial(settings, ``exception``) = printfn "%A" ``exception``
                 member this.ReportRunningTrial(setting) = s.Value <- setting            
            })

let genomeSize = Grammar.estimateGenomeSize g + 1
let oPl,oAcc,rlst,cache = Optimize.run genomeSize 20000 CA.OptimizationKind.Minimize (expFac (10u * 60u)) g

Grammar.printPipeline ctx oPl
rlst.TrialSettings