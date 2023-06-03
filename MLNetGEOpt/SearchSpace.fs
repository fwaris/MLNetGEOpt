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
    let inline zeroIsDefault<'a when 'a: equality> (v:'a) = if v = Unchecked.defaultof<'a> then None else Some v
    (*
    zeroIsDefault 0.f
    zeroIsDefault 0
    zeroIsDefault 0.0
    zeroIsDefault 1.0    
    *)
    let PIPELINE = "_pipeline_"
    let FEATURES = "Features"

    let seNorm (col:string) (mbcMin,mbcMax) () =         
        let lF = "fixZero"
        let lM = "maximumBinCount"
        let fac (ctx:MLContext) (p:Parameter) = 
            let fixZero = p.[lF].AsType<bool>()
            let mbc = p.[lM].AsType<int>() 
            let mbc = if mbc = 1 then None else Some(mbc)
            ctx.Transforms.NormalizeBinning(col,fixZero=fixZero,?maximumBinCount=mbc) |> asEstimator        
        let ss = Search.init() |> Search.withChoice(lF,[|true;false|]) |> Search.withUniformInt(lM,mbcMin,mbcMax)
        SweepableEstimator(fac,ss)

    let seGlobalContrast (col:string) (sclMin,sclMax) () =         
        let lZm = "ensureZeroMean"
        let lSd = "ensureUnitStandardDeviation"
        let lScl = "scale"
        let fac (ctx:MLContext) (p:Parameter) = 
            let ezm = p.[lZm].AsType<bool>()
            let eusd = p.[lSd].AsType<bool>() 
            let scale = p.[lScl].AsType<float32>() |> zeroIsDefault
            ctx.Transforms.NormalizeGlobalContrast(col,ensureZeroMean=ezm,ensureUnitStandardDeviation=eusd,?scale=scale) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withChoice(lZm,[|true;false|]) 
            |> Search.withChoice(lSd,[|true;false|])
            |> Search.withUniformFloat32(lScl,sclMin,sclMax)
        SweepableEstimator(fac,ss)

    let seNormLogMeanVar (col:string) () =     
        let lucf = "useCdf"
        let fac (ctx:MLContext) (p:Parameter) = 
            let useCdf = p.[lucf].AsType<bool>()
            ctx.Transforms.NormalizeLogMeanVariance(col,useCdf=useCdf) |> asEstimator
        let ss = Search.init() |> Search.withChoice(lucf,[|true;false|]) 
        SweepableEstimator(fac,ss)

    let seNormLpNorm (col:string) () = 
        let lEzm = "ensureZeroMean"
        let lNorm = "norm"
        let fac (ctx:MLContext) (p:Parameter) = 
            let ezm = p.[lEzm].AsType<bool>()
            let norm = p.[lNorm].AsType<Microsoft.ML.Transforms.LpNormNormalizingEstimatorBase.NormFunction>()
            ctx.Transforms.NormalizeLpNorm(col,norm=norm,ensureZeroMean=ezm) |> asEstimator
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

    let seNormRobustScaling (col:string) () =       
        let lcntr = "centerData"
        let fac (ctx:MLContext) (p:Parameter) = 
            let centerData = p.[lcntr].AsType<bool>()
            ctx.Transforms.NormalizeRobustScaling(col,centerData=centerData) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withChoice(lcntr,[|true;false|]) 
        SweepableEstimator(fac,ss)
    
    let seNormMinMax (col:string) ()=
        let lfixz = "fixZero"
        let fac (ctx:MLContext) (p:Parameter) = 
            let fixZero = p.[lfixz].AsType<bool>()
            ctx.Transforms.NormalizeMinMax("Features",fixZero=fixZero) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withChoice(lfixz,[|true;false|]) 
        SweepableEstimator(fac,ss)

    let seNormSupBin (col:string)  label () =
        let lfixz = "fixZero"
        let fac (ctx:MLContext) (p:Parameter) = 
            let fixZero = p.[lfixz].AsType<bool>()
            ctx.Transforms.NormalizeSupervisedBinning(col,fixZero=fixZero,labelColumnName=label) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withChoice(lfixz,[|true;false|]) 
        SweepableEstimator(fac,ss)

    let seWhiten (col:string) () = 
        let lkind = "WhiteningKind"
        let lrank = "rank"
        let fac (ctx:MLContext) (p:Parameter) =         
            let kind = p.[lkind].AsType<Microsoft.ML.Transforms.WhiteningKind>()
            let rank = p.[lrank].AsType<int>() |> zeroIsDefault
            ctx.Transforms.VectorWhiten(col,kind=kind,?rank=rank) |> asEstimator
        let wvals = 
            Enum.GetValues(typeof<Transforms.WhiteningKind>) 
            |> Seq.cast<Transforms.WhiteningKind>
            |> Seq.map box
            |> Seq.toArray
        let ss = 
            Search.init() 
            |> Search.withChoice(lkind,wvals) 
            |> Search.withUniformInt(lrank,0,5)
        SweepableEstimator(fac,ss)

    let seNormMeanVar (col:string) ()= 
        let lucf = "useCdf"
        let lEzm = "ensureZeroMean"
        let fac (ctx:MLContext) (p:Parameter) = 
            let useCdf = p.[lucf].AsType<bool>()
            let ezm = p.[lEzm].AsType<bool>()
            ctx.Transforms.NormalizeMeanVariance(col,useCdf=useCdf,fixZero=ezm) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withChoice(lucf,[|true;false|]) 
            |> Search.withChoice(lEzm,[|true;false|])
        SweepableEstimator(fac,ss)

    let seProjPca (col:string) (rLo,rHi) () =
        let lrank = "rank"
        let lovsmp = "overSampling"
        let lezm = "ensureZeroMean"
        let fac (ctx:MLContext) (p:Parameter) =
            let rank = p.[lrank].AsType<int>() |> zeroIsDefault
            let ensureZeroMean = p.[lezm].AsType<bool>()
            ctx.Transforms.ProjectToPrincipalComponents(col,?rank=rank,ensureZeroMean=ensureZeroMean) |> asEstimator
        let ss =
            Search.init()
            |> Search.withChoice(lezm,[|true;false|])
            |> Search.withUniformInt(lrank,rLo,rHi)
        SweepableEstimator(fac, ss)

    let [<Literal>] private Kgaus = "gaussian"
    let [<Literal>] private Klap = "laplacian"
    let seKernelMap (col:string) (rLo,rHi) () =
        let lrank = "rank"
        let lcossin = "useCosAndSinBases"
        let lgen = "generator"
        let fac (ctx:MLContext) (p:Parameter) =
            let rank = p.[lrank].AsType<int>() |> zeroIsDefault
            let useCosAndSinBases = p.[lcossin].AsType<bool>() 
            let generator = p.[lgen].AsType<string>()
            let generator  = 
                match generator with 
                | Kgaus -> Transforms.GaussianKernel()  :> Transforms.KernelBase |> Some 
                | Klap  -> Transforms.LaplacianKernel() :> Transforms.KernelBase |> Some 
                | _     -> None
            ctx.Transforms.ApproximatedKernelMap(col,?rank=rank,useCosAndSinBases=useCosAndSinBases,?generator=generator) |> asEstimator
        let ss =
            Search.init()
            |> Search.withChoice(lcossin,[|true;false|])
            |> Search.withUniformInt(lrank,rLo,rHi)
            |> Search.withChoice(lgen,[|Kgaus; Klap|])
        SweepableEstimator(fac, ss)

    let seFtrSelCount (col:string) count () =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.FeatureSelection.SelectFeaturesBasedOnCount(col,count=count) |> asEstimator
        let ss =
            Search.init()
        SweepableEstimator(fac, ss)

    let seFtrSelMutualInf (col:string) label () =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation(col,labelColumnName=label) |> asEstimator
        let ss =
            Search.init()
        SweepableEstimator(fac, ss)

    let seMissingVals (col:string) () =
        let lrmode = "replacementMode"
        let limpt = "imputeBySlot"
        let fac (ctx:MLContext) (p:Parameter) =
            let replacementMode = p.[lrmode].AsType<Transforms.MissingValueReplacingEstimator.ReplacementMode>()
            let imputeBySlot = p.[limpt].AsType<bool>()
            ctx.Transforms.ReplaceMissingValues(col,replacementMode=replacementMode,imputeBySlot=imputeBySlot) |> asEstimator
        let wvals = 
            Enum.GetValues(typeof<Transforms.MissingValueReplacingEstimator.ReplacementMode>) 
            |> Seq.cast<Transforms.MissingValueReplacingEstimator.ReplacementMode>
            |> Seq.map box
            |> Seq.toArray
        let ss =
            Search.init()
            |> Search.withChoice(lrmode,wvals)
            |> Search.withChoice(limpt,[|true;false|])            
        SweepableEstimator(fac, ss)

    let seTextFeaturize (txtCol:string) () =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.Text.FeaturizeText(txtCol) |> asEstimator
        let ss =
            Search.init()
        SweepableEstimator(fac, ss)

    let seTextHashedNGrams (txtCol:string) () =
        let lbits = "numberOfBits"
        let lngl = "ngramLength"
        let lskl = "skipLength"
        let lalll = "useAllLengths"
        let loh = "useOrderedHashing"
        let fac (ctx:MLContext) (p:Parameter) =
            let numberOfBits = p.[lbits].AsType<int>() |> zeroIsDefault
            let ngramLength = p.[lngl].AsType<int>()  |> zeroIsDefault
            let skipLength = p.[lskl].AsType<int>()   |> zeroIsDefault
            let useAllLengths = p.[lalll].AsType<bool>()
            let useOrderedHashing = p.[loh].AsType<bool>()
            let tx1 = ctx.Transforms.Text.TokenizeIntoCharactersAsKeys("tokens",txtCol)
            let tx2 = ctx.Transforms.Text.ProduceHashedNgrams(txtCol,"tokens",?numberOfBits=numberOfBits,?ngramLength=ngramLength,
                                                    ?skipLength=skipLength,useAllLengths=useAllLengths,
                                                    useOrderedHashing=useOrderedHashing)
            tx1 <!> tx2
        let ss =
            Search.init()
            |> Search.withUniformInt(lbits,0,4)
            |> Search.withUniformInt(lngl,0,5)
            |> Search.withUniformInt(lskl,0,5)
            |> Search.withChoice(lalll,[|true;false|])
            |> Search.withChoice(loh,[|true;false|])
        SweepableEstimator(fac, ss)

    ///search terms with input col defaulted to FEATURES 
    module Def =
        let seNorm = seNorm FEATURES
        let seGlobalContrast = seGlobalContrast FEATURES
        let seNormLogMeanVar = seNormLogMeanVar FEATURES
        let seNormLpNorm = seNormLpNorm FEATURES
        let seNormRobustScaling = seNormRobustScaling FEATURES
        let seNormMinMax = seNormMinMax FEATURES
        let seNormSupBin = seNormSupBin FEATURES
        let seWhiten = seWhiten FEATURES
        let seNormMeanVar = seNormMeanVar FEATURES
        let seProjPca = seProjPca FEATURES
        let seKernelMap = seKernelMap FEATURES
        let seFtrSelCount = seFtrSelCount FEATURES
        let seFtrSelMutualInf = seFtrSelMutualInf FEATURES
        let seMissingVals = seMissingVals FEATURES

