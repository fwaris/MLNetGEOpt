namespace MLNetGEOpt
open System
open Microsoft.ML.AutoML
open Microsoft.ML.SearchSpace
open Microsoft.ML
open MLUtils.Pipeline

type Search =
    static member TERM_ID = "_term_id_"
    static member init() = new SearchSpace()
    static member withUniformFloat(s:string,lo,hi,?logBase,?defaultValue)  = fun (ss:SearchSpace) -> ss.Add(s,Option.UniformDoubleOption(lo,hi,?logBase=logBase,?defaultValue=defaultValue)); ss
    static member withUniformInt (s:string,lo,hi,?logBase,?defaultValue) =  fun (ss:SearchSpace) -> ss.Add(s,Option.UniformIntOption(lo,hi,?logBase=logBase,?defaultValue=defaultValue)); ss
    static member withUniformFloat32 (s:string,lo,hi,?logBase,?defaultValue) = fun (ss:SearchSpace) -> ss.Add(s,Option.UniformSingleOption(lo,hi,?logBase=logBase,?defaultValue=defaultValue)); ss
    static member withChoice(s:string,choices: obj[],?defaultChoice) = fun  (ss:SearchSpace) -> let opt = match defaultChoice with Some d -> Option.ChoiceOption(choices,d) | _-> Option.ChoiceOption(choices) in ss.Add(s,opt); ss
    static member withId(s:string) = Search.withChoice(Search.TERM_ID,[|s|])

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

    let seNormBin (col:string) (mbcMin,mbcMax) () =         
        let lF = "fixZero"
        let lM = "maximumBinCount"
        let fac (ctx:MLContext) (p:Parameter) = 
            let fixZero = p.[lF].AsType<bool>()
            let mbc = p.[lM].AsType<int>() 
            let mbc = if mbc = 1 then None else Some(mbc)
            printfn $"seNormBin {col} {lF}={fixZero}, {1M}={mbc}"
            ctx.Transforms.NormalizeBinning(col,fixZero=fixZero,?maximumBinCount=mbc) |> asEstimator        
        let ss = 
            Search.init() 
            |> Search.withId $"seNormBin {col}"
            |> Search.withChoice(lF,[|true;false|]) 
            |> Search.withUniformInt(lM,mbcMin,mbcMax)
        SweepableEstimator(fac,ss)

    let seGlobalContrast (col:string) (sclMin,sclMax) () =         
        let lZm = "ensureZeroMean"
        let lSd = "ensureUnitStandardDeviation"
        let lScl = "scale"
        let fac (ctx:MLContext) (p:Parameter) = 
            let ezm = p.[lZm].AsType<bool>()
            let eusd = p.[lSd].AsType<bool>() 
            let scale = p.[lScl].AsType<float32>() |> zeroIsDefault
            printfn $"seGlobalContrast {col} {lZm}={ezm}, {lSd}={eusd}, {lScl}={scale}"
            ctx.Transforms.NormalizeGlobalContrast(col,ensureZeroMean=ezm,ensureUnitStandardDeviation=eusd,?scale=scale) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seGlobalContrast {col}"
            |> Search.withChoice(lZm,[|true;false|]) 
            |> Search.withChoice(lSd,[|true;false|])
            |> Search.withUniformFloat32(lScl,sclMin,sclMax)
        SweepableEstimator(fac,ss)

    let seNormLogMeanVar (col:string) () =     
        let lucf = "useCdf"
        let fac (ctx:MLContext) (p:Parameter) = 
            let useCdf = p.[lucf].AsType<bool>()
            printfn $"seNormLogMeanVar {col} {lucf}={useCdf}"
            ctx.Transforms.NormalizeLogMeanVariance(col,useCdf=useCdf) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seNormLogMeanVar {col}"
            |> Search.withChoice(lucf,[|true;false|]) 
        SweepableEstimator(fac,ss)

    let seNormLpNorm (col:string) () = 
        let lEzm = "ensureZeroMean"
        let lNorm = "norm"
        let fac (ctx:MLContext) (p:Parameter) = 
            let ezm = p.[lEzm].AsType<bool>()
            let norm = p.[lNorm].AsType<Microsoft.ML.Transforms.LpNormNormalizingEstimatorBase.NormFunction>()
            printfn $"seNormLpNorm {col} {lEzm}={ezm}, {lNorm}={norm}"
            ctx.Transforms.NormalizeLpNorm(col,norm=norm,ensureZeroMean=ezm) |> asEstimator
        let normVals = 
            Enum.GetValues(typeof<Microsoft.ML.Transforms.LpNormNormalizingEstimatorBase.NormFunction>) 
            |> Seq.cast<Microsoft.ML.Transforms.LpNormNormalizingEstimatorBase.NormFunction>
            |> Seq.map box
            |> Seq.toArray
        let ss = 
            Search.init() 
            |> Search.withId $"seNormLpNorm {col}"
            |> Search.withChoice(lEzm,[|true;false|]) 
            |> Search.withChoice(lNorm, normVals)
        SweepableEstimator(fac,ss)

    let seNormRobustScaling (col:string) () =       
        let lcntr = "centerData"
        let fac (ctx:MLContext) (p:Parameter) = 
            let centerData = p.[lcntr].AsType<bool>()
            printfn $"seNormRobustScaling {col} {lcntr}={centerData}"
            ctx.Transforms.NormalizeRobustScaling(col,centerData=centerData) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seNormRobustScaling {col}"
            |> Search.withChoice(lcntr,[|true;false|]) 
        SweepableEstimator(fac,ss)
    
    let seNormMinMax (col:string) ()=
        let lfixz = "fixZero"
        let fac (ctx:MLContext) (p:Parameter) = 
            let fixZero = p.[lfixz].AsType<bool>()
            printfn $"seNormMinMax {col} {lfixz}={fixZero}"
            ctx.Transforms.NormalizeMinMax(col,fixZero=fixZero) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seNormMinMax {col}"
            |> Search.withChoice(lfixz,[|true;false|]) 
        SweepableEstimator(fac,ss)

    let seNormSupBin (col:string)  label () =
        let lfixz = "fixZero"
        let fac (ctx:MLContext) (p:Parameter) = 
            let fixZero = p.[lfixz].AsType<bool>()
            printfn $"seNormSupBin {col} {lfixz}={fixZero}"
            ctx.Transforms.NormalizeSupervisedBinning(col,fixZero=fixZero,labelColumnName=label) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seNormSupBin {col}"
            |> Search.withChoice(lfixz,[|true;false|]) 
        SweepableEstimator(fac,ss)

    let seWhiten (col:string) () = 
        let lkind = "WhiteningKind"
        let lrank = "rank"
        let fac (ctx:MLContext) (p:Parameter) =         
            let kind = p.[lkind].AsType<Microsoft.ML.Transforms.WhiteningKind>()
            let rank = p.[lrank].AsType<int>()
            let rank = if kind = Transforms.WhiteningKind.PrincipalComponentAnalysis then Some rank else None
            printfn $"seWhiten {col} {lkind}={kind}, {lrank}={rank}"
            ctx.Transforms.VectorWhiten(col,kind=kind,?rank=rank) |> asEstimator
        let wvals = 
            Enum.GetValues(typeof<Transforms.WhiteningKind>) 
            |> Seq.cast<Transforms.WhiteningKind>
            |> Seq.map box
            |> Seq.toArray
        let ss = 
            Search.init() 
            |> Search.withId $"seWhiten {col}"
            |> Search.withChoice(lkind,wvals) 
            |> Search.withUniformInt(lrank,2,5)
        SweepableEstimator(fac,ss)

    let seNormMeanVar (col:string) ()= 
        let lucf = "useCdf"
        let lEzm = "ensureZeroMean"
        let fac (ctx:MLContext) (p:Parameter) = 
            let useCdf = p.[lucf].AsType<bool>()
            let ezm = p.[lEzm].AsType<bool>()
            printfn $"seNormMeanVar {col} {lucf}={useCdf}, {lEzm}={ezm}"
            ctx.Transforms.NormalizeMeanVariance(col,useCdf=useCdf,fixZero=ezm) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seNormMeanVar {col}"
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
            |> Search.withId $"seProjPca {col}"
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
            |> Search.withId $"seKernelMap {col}"
            |> Search.withChoice(lcossin,[|true;false|])
            |> Search.withUniformInt(lrank,rLo,rHi)
            |> Search.withChoice(lgen,[|Kgaus; Klap|])
        SweepableEstimator(fac, ss)

    let seFtrSelCount (col:string) count () =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.FeatureSelection.SelectFeaturesBasedOnCount(col,count=count) |> asEstimator
        let ss =
            Search.init()
            |> Search.withId $"seFtrSelCount {col}"
        SweepableEstimator(fac, ss)

    let seFtrSelMutualInf (col:string) label () =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation(col,labelColumnName=label) |> asEstimator
        let ss =
            Search.init()
            |> Search.withId $"seFtrSelMutualInf {col}"
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
            |> Search.withId $"seMissingVals {col}"
            |> Search.withChoice(lrmode,wvals)
            |> Search.withChoice(limpt,[|true;false|])            
        SweepableEstimator(fac, ss)

    let seTextFeaturize (col:string) () =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.Text.FeaturizeText(col) |> asEstimator
        let ss =
            Search.init()
            |> Search.withId $"seTextFeaturize {col}"
        SweepableEstimator(fac, ss)

    let seTextHashedNGrams (col:string) () =
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
            let tx1 = ctx.Transforms.Text.TokenizeIntoCharactersAsKeys("tokens",col)
            let tx2 = ctx.Transforms.Text.ProduceHashedNgrams(col,"tokens",?numberOfBits=numberOfBits,?ngramLength=ngramLength,
                                                    ?skipLength=skipLength,useAllLengths=useAllLengths,
                                                    useOrderedHashing=useOrderedHashing)
            tx1 <!> tx2
        let ss =
            Search.init()
            |> Search.withId $"seTextHashedNGrams {col}"
            |> Search.withUniformInt(lbits,0,4)
            |> Search.withUniformInt(lngl,0,5)
            |> Search.withUniformInt(lskl,0,5)
            |> Search.withChoice(lalll,[|true;false|])
            |> Search.withChoice(loh,[|true;false|])
        SweepableEstimator(fac, ss)

///Module where search space is part of the grammar terms (avoids issue that ML.Net is not exploring search space well)
[<RequireQualifiedAccess>]
module Eh =
    (*
    zeroIsDefault 0.f
    zeroIsDefault 0
    zeroIsDefault 0.0
    zeroIsDefault 1.0    
    *)

    let seNormBin (col:string) fixZero maximumBinCount () =         
        let fac (ctx:MLContext) (p:Parameter) = 
            ctx.Transforms.NormalizeBinning(col,fixZero=fixZero,?maximumBinCount=maximumBinCount) |> asEstimator        
        let ss = 
            Search.init() 
            |> Search.withId $"seNormBin {col} fixZero={fixZero} maximumBinCount={maximumBinCount}"
        SweepableEstimator(fac,ss)

    let seGlobalContrast (col:string) ensureZeroMean ensureUnitStandardDeviation scale () =         
        let fac (ctx:MLContext) (p:Parameter) = 
            ctx.Transforms.NormalizeGlobalContrast(col,ensureZeroMean=ensureZeroMean,ensureUnitStandardDeviation=ensureUnitStandardDeviation,?scale=scale) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seGlobalContrast {col} ensureZeroMean={ensureZeroMean} ensureUnitStandardDeviation={ensureUnitStandardDeviation} scale={scale}"
        SweepableEstimator(fac,ss)

    let seNormLogMeanVar (col:string) useCdf () =             
        let fac (ctx:MLContext) (p:Parameter) = 
            ctx.Transforms.NormalizeLogMeanVariance(col,useCdf=useCdf) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seNormLogMeanVar {col} useCdf={useCdf}"
        SweepableEstimator(fac,ss)

    let SENormLpNorm_Norms() = 
        [
            yield! Enum.GetValues(typeof<Transforms.LpNormNormalizingEstimatorBase.NormFunction>) 
            |> Seq.cast<Transforms.LpNormNormalizingEstimatorBase.NormFunction>
            |> Seq.map Some

        ]
    let seNormLpNorm (col:string) norm ensureZeroMean () = 
        let fac (ctx:MLContext) (p:Parameter) = 
            ctx.Transforms.NormalizeLpNorm(col,?norm=norm,ensureZeroMean=ensureZeroMean) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seNormLpNorm {col} norm={norm} ensureZeroMean={ensureZeroMean}"
        SweepableEstimator(fac,ss)

    let seNormRobustScaling (col:string) centerData () =           
        let fac (ctx:MLContext) (p:Parameter) = 
            ctx.Transforms.NormalizeRobustScaling(col,centerData=centerData) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seNormRobustScaling {col} centerData={centerData}"
        SweepableEstimator(fac,ss)
    
    let seNormMinMax (col:string) fixZero ()=
        let fac (ctx:MLContext) (p:Parameter) = 
            ctx.Transforms.NormalizeMinMax(col,fixZero=fixZero) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seNormMinMax {col} fixZero={fixZero}"
        SweepableEstimator(fac,ss)

    let seNormSupBin (col:string) labelColumnName fixZero maximumBinCount () =        
        let fac (ctx:MLContext) (p:Parameter) = 
            ctx.Transforms.NormalizeSupervisedBinning(col,?fixZero=fixZero, ?maximumBinCount=maximumBinCount,labelColumnName=labelColumnName) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seNormSupBin {col} labelColumnName={labelColumnName} fixZero={fixZero} maximumBinCount={maximumBinCount}"            
        SweepableEstimator(fac,ss)

    let SEWhiten_Kinds() =
        [
            yield! Enum.GetValues(typeof<Transforms.WhiteningKind>) 
            |> Seq.cast<Transforms.WhiteningKind>
            |> Seq.map Some

        ]    
    let seWhiten (col:string) kind rank () = 
        let fac (ctx:MLContext) (p:Parameter) =         
            ctx.Transforms.VectorWhiten(col,?kind=kind,?rank=rank) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seWhiten {col} kind={kind} rank={rank}"
        SweepableEstimator(fac,ss)

    let seNormMeanVar (col:string) fixZero useCdf ()= 
        let fac (ctx:MLContext) (p:Parameter) = 
            ctx.Transforms.NormalizeMeanVariance(col,fixZero=fixZero,useCdf=useCdf) |> asEstimator
        let ss = 
            Search.init() 
            |> Search.withId $"seNormMeanVar {col} fixZero={fixZero} useCdf={useCdf}"
        SweepableEstimator(fac,ss)

    let seProjPca (col:string) rank ensureZeroMean () =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.ProjectToPrincipalComponents(col,?rank=rank,?ensureZeroMean=ensureZeroMean) |> asEstimator
        let ss =
            Search.init()
            |> Search.withId $"seProjPca {col} rank={rank} ensureZeroMean={ensureZeroMean}"
        SweepableEstimator(fac, ss)

    let SEKernelMap_Generators() = 
        [
            Transforms.GaussianKernel() :> Transforms.KernelBase |> Some
            Transforms.LaplacianKernel() :> Transforms.KernelBase |> Some
        ]
    let seKernelMap (col:string) rank useCosAndSinBases generator () =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.ApproximatedKernelMap(col,?rank=rank,useCosAndSinBases=useCosAndSinBases,?generator=generator) |> asEstimator
        let ss =
            Search.init()
            |> Search.withId $"seKernelMap {col} rank={rank} useCosAndSinBases={useCosAndSinBases} generator={generator}"
        SweepableEstimator(fac, ss)

    let seFtrSelCount (col:string) count () =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.FeatureSelection.SelectFeaturesBasedOnCount(col,count=count) |> asEstimator
        let ss =
            Search.init()
            |> Search.withId $"seFtrSelCount {col} count={count}"
        SweepableEstimator(fac, ss)

    let seFtrSelMutualInf (col:string) labelColumnName () =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation(col,labelColumnName=labelColumnName) |> asEstimator
        let ss =
            Search.init()
            |> Search.withId $"seFtrSelMutualInf {col} labelColumnName={labelColumnName}"
        SweepableEstimator(fac, ss)

    let SEMissingVals_ReplacementModes() = 
        [            
            yield! Enum.GetValues(typeof<Transforms.MissingValueReplacingEstimator.ReplacementMode>) 
            |> Seq.cast<Transforms.MissingValueReplacingEstimator.ReplacementMode>
            |> Seq.map Some

        ]
    let seMissingVals (col:string) replacementMode imputeBySlot () =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.ReplaceMissingValues(col,?replacementMode=replacementMode,imputeBySlot=imputeBySlot) |> asEstimator
        let ss =
            Search.init()
            |> Search.withId $"seMissingVals {col} replacementMode={replacementMode} imputeBySlot={imputeBySlot}"
        SweepableEstimator(fac, ss)

    let seTextFeaturize (col:string) () =
        let fac (ctx:MLContext) (p:Parameter) =
            ctx.Transforms.Text.FeaturizeText(col) |> asEstimator
        let ss =
            Search.init()
            |> Search.withId $"seTextFeaturize {col}"
        SweepableEstimator(fac, ss)

    let seTextHashedNGrams (col:string) numberOfBits ngramLength skipLength useAllLengths useOrderedHashing () =
        let fac (ctx:MLContext) (p:Parameter) =
            let tx1 = ctx.Transforms.Text.TokenizeIntoCharactersAsKeys("tokens",col)
            let tx2 = ctx.Transforms.Text.ProduceHashedNgrams(col,"tokens",?numberOfBits=numberOfBits,?ngramLength=ngramLength,
                                                    ?skipLength=skipLength,useAllLengths=useAllLengths,
                                                    useOrderedHashing=useOrderedHashing)
            tx1 <!> tx2
        let ss =
            Search.init()
            |> Search.withId $"seTextHashedNGrams {col} numberOfBits={numberOfBits} ngramLength={ngramLength} skipLength={skipLength} useAllLengths={useAllLengths} useOrderedHashing={useOrderedHashing}"
        SweepableEstimator(fac, ss)

    ///search terms with input col defaulted to FEATURES 
    module Def =
        let seNormBin = seNormBin E.FEATURES
        let seGlobalContrast = seGlobalContrast E.FEATURES
        let seNormLogMeanVar = seNormLogMeanVar E.FEATURES
        let seNormLpNorm = seNormLpNorm E.FEATURES
        let seNormRobustScaling = seNormRobustScaling E.FEATURES
        let seNormMinMax = seNormMinMax E.FEATURES
        let seNormSupBin = seNormSupBin E.FEATURES
        let seWhiten = seWhiten E.FEATURES
        let seNormMeanVar = seNormMeanVar E.FEATURES
        let seProjPca = seProjPca E.FEATURES
        let seKernelMap = seKernelMap E.FEATURES
        let seFtrSelCount = seFtrSelCount E.FEATURES
        let seFtrSelMutualInf = seFtrSelMutualInf E.FEATURES
        let seMissingVals = seMissingVals E.FEATURES

