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
        "cat2"
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

let seCluster () =         
    let K = "k"
    let fac (ctx:MLContext) (p:Parameter) = 
        let k = p.[K].AsType<int>()
        let opts = Trainers.KMeansTrainer.Options()
        opts.FeatureColumnName <- E.FEATURES
        opts.MaximumNumberOfIterations <- 200
        opts.NumberOfClusters <- k    
        ctx.Clustering.Trainers.KMeans(opts) |> asEstimator              
    let ss = 
        Search.init() 
        |> Search.withId $"Cluster size"
        |> Search.withUniformInt(K,3,10,defaultValue=5)
    SweepableEstimator(fac,ss)

let g = 
    [
        Estimator seBase
        //Pipeline featurizer
        Opt(Estimator (E.Def.seFtrSelCount 3))
        Opt(
            Alt [
                Alt ([(1,10); (11,20); (21,30); (31,100)] |> List.map(E.Def.seNorm>>Estimator))
                Estimator E.Def.seNormLpNorm
                Estimator E.Def.seNormLogMeanVar
                Estimator E.Def.seNormMeanVar
                Alt([0.1f .. 0.5f .. 4.0f] |> List.pairwise |> List.map(fun (a,b) -> a, b - 0.001f)  |> List.map(E.Def.seGlobalContrast>>Estimator))
                Estimator E.Def.seNormMinMax
                Estimator E.Def.seNormRobustScaling
            ])
        //Opt (Estimator E.Def.seWhiten)
        //Opt (
        //    Alt [
        //        Estimator (E.Def.seProjPca (2,10))
        //        Estimator (E.Def.seKernelMap (2,10))
        //    ])
        Estimator seCluster
    ]

let runTrial (pipeline:SweepablePipeline) (settings:TrialSettings) =     
    task {
        let stopwatch = Stopwatch()
        let parameter = settings.Parameter[E.PIPELINE]
        // Use parameters to build pipeline
        let pipeline = pipeline.BuildFromOption(ctx, parameter)
        let model = pipeline.Fit(dvTrain)
        let scored = model.Transform(dvTest)
        let eval = ctx.Clustering.Evaluate(scored)
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
            let printLine (isDone:bool) (r:TrialResult) =  printfn $"""M: {r.Metric} {isDone} - {s.Value.Parameter.[E.PIPELINE]}"""
            {new IMonitor with
                 member this.ReportBestTrial(result) = printLine true result 
                 member this.ReportCompletedTrial(result) = printLine false result 
                 member this.ReportFailTrial(settings, ``exception``) = printfn "%A" ``exception``
                 member this.ReportRunningTrial(setting) = s.Value <- setting            
            })

//let genomeSize = Grammar.estimateGenomeSize g
let oPl,oAcc,rlst,cache = Optimize.run 11 15000 CA.OptimizationKind.Minimize (expFac 60u) g
Grammar.printPipeline ctx oPl




