namespace MLNetGEOpt
open System
open Microsoft.ML.AutoML
open Microsoft.ML.SearchSpace
open Microsoft.ML
open MLUtils.Pipeline

type Search =
    static member init() = new SearchSpace()
    static member withUniformFloat(s:string,lo,hi,?logBase,?defaultValue)  = fun (ss:SearchSpace) -> ss.Add(s,Option.UniformDoubleOption(lo,hi,?logBase=logBase,?defaultValue=defaultValue)); ss
    static member withUniformInt (s:string,lo,hi,?logBase,?defaultValue) =  fun (ss:SearchSpace) -> ss.Add(s,Option.UniformIntOption(lo,hi,?logBase=logBase,?defaultValue=defaultValue)); ss
    static member withUniformFloat32 (s:string,lo,hi,?logBase,?defaultValue) = fun (ss:SearchSpace) -> ss.Add(s,Option.UniformSingleOption(lo,hi,?logBase=logBase,?defaultValue=defaultValue)); ss
    static member withChoice(s:string,choices: obj[],?defaultChoice) = fun  (ss:SearchSpace) -> let opt = match defaultChoice with Some d -> Option.ChoiceOption(choices,d) | _-> Option.ChoiceOption(choices) in ss.Add(s,opt); ss

[<RequireQualifiedAccess>]
module E =
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

    let seNormRobustScaling =
        let lcntr = "centerData"
        let fac (ctx:MLContext) (p:Parameter) = 
            let centerData = p.[lcntr].AsType<bool>()
            ctx.Transforms.NormalizeRobustScaling("Features",centerData=centerData) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withChoice(lcntr,[|true;false|]) 
        SweepableEstimator(fac,ss)
    
    let seNormMinMax =
        let lfixz = "fixZero"
        let fac (ctx:MLContext) (p:Parameter) = 
            let fixZero = p.[lfixz].AsType<bool>()
            ctx.Transforms.NormalizeMinMax("Features",fixZero=fixZero) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withChoice(lfixz,[|true;false|]) 
        SweepableEstimator(fac,ss)

    let seNormSupBin label =
        let lfixz = "fixZero"
        let fac (ctx:MLContext) (p:Parameter) = 
            let fixZero = p.[lfixz].AsType<bool>()
            ctx.Transforms.NormalizeSupervisedBinning("Features",fixZero=fixZero,labelColumnName=label) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withChoice(lfixz,[|true;false|]) 
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

    let seProjPca (rLo,rHi) =
        let lrank = "rank"
        let lovsmp = "overSampling"
        let lezm = "ensureZeroMean"
        let fac (ctx:MLContext) (p:Parameter) =
            let rank = p.[lrank].AsType<int>()
            let rank = if rank = 0 then None else Some rank
            let ensureZeroMean = p.[lezm].AsType<bool>()
            ctx.Transforms.ProjectToPrincipalComponents("Features",?rank=rank,ensureZeroMean=ensureZeroMean) |> asEstimator
        let ss =
            Search.init()
            |> Search.withChoice(lezm,[|true;false|])
            |> Search.withUniformInt(lrank,rLo,rHi)
        SweepableEstimator(fac, ss)

    let [<Literal>] private Kgaus = "gaussian"
    let [<Literal>] private Klap = "laplacian"
    let seKernelMap (rLo,rHi) =
        let lrank = "rank"
        let lcossin = "useCosAndSinBases"
        let lgen = "generator"
        let fac (ctx:MLContext) (p:Parameter) =
            let rank = p.[lrank].AsType<int>()
            let rank = if rank = 0 then None else Some rank
            let useCosAndSinBases = p.[lcossin].AsType<bool>() 
            let generator = p.[lgen].AsType<string>()
            let generator  = 
                match generator with 
                | Kgaus -> Transforms.GaussianKernel()  :> Transforms.KernelBase |> Some 
                | Klap  -> Transforms.LaplacianKernel() :> Transforms.KernelBase |> Some 
                | _     -> None
            ctx.Transforms.ApproximatedKernelMap("Features",?rank=rank,useCosAndSinBases=useCosAndSinBases,?generator=generator) |> asEstimator
        let ss =
            Search.init()
            |> Search.withChoice(lcossin,[|true;false|])
            |> Search.withUniformInt(lrank,rLo,rHi)
            |> Search.withChoice(lgen,[|Kgaus; Klap|])
        SweepableEstimator(fac, ss)

    let seFtrSelCount count =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.FeatureSelection.SelectFeaturesBasedOnCount("Features",count=count) |> asEstimator
        let ss =
            Search.init()
        SweepableEstimator(fac, ss)

    let seFtrSelMutualInf label =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation("Features",labelColumnName=label) |> asEstimator
        let ss =
            Search.init()
        SweepableEstimator(fac, ss)

