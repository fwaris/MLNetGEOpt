#load "packages.fsx"
open System
open System.IO
open FSharp.Collections.ParallelSeq
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.AutoML
open Microsoft.ML.SearchSpace
open MLNetGEOpt
open MLUtils
open MLUtils.Pipeline
open MathNet.Numerics
open Plotly.NET

let root = @"E:\s\kaggle\predict-student-performance-from-game-play"
let (@@) a b = Path.Combine(a,b)

let trainData = root @@ "train.csv"
let trainLables = root @@ "train_labels.csv"
let dvFile =  root @@ "baseSet.dv"

[<CLIMutable>]
type SEventAgg = 
    {
        session_id: Single
        level_group: string // Key<UInt32, 0-2> {KeyValues}
        count : single
        elapsed_time_mean: Single
        elapsed_tiem_std : single
        event_name_count: Single
        name_count: Single; //Key<UInt32, 0-5> {KeyValues}
        room_coor_x_mean: Single
        room_coor_x_std: Single
        room_coor_y_mean: Single
        room_coor_y_std: Single
        screen_coor_x_mean: Single
        screen_coor_x_std: Single
        screen_coor_y_mean: Single
        screen_coor_y_std: Single
        hover_duration_mean: Single
        hover_duration_std: Single
        text: string
        fqid_count: single // uint32 // Key<UInt32, 0-127> {KeyValues}
        room_fqid_count: single // uint32 // Key<UInt32, 0-18> {KeyValues}
        text_fqid_count: single // uint32 ///Key<UInt32, 0-125> {KeyValues}
        fullscreen_count: single // uint32 // Key<UInt32, 0-1> {KeyValues}
        hq_count : single //uint32 // Key<UInt32, 0-1> {KeyValues}
        music_count : single //uint32 // Key<UInt32, 0-1> {KeyValues}
        answers : single[]
    }

let ctx = MLContext()

module Featurize =
    [<CLIMutable>]
    type SEvent = {
        session_id : Single
        index : Single
        elapsed_time : Single
        event_name : UInt32
        name : UInt32
        level : UInt32
        page : UInt32
        room_coor_x : Single
        room_coor_y : Single
        screen_coor_x : Single
        screen_coor_y : Single
        hover_duration : Single
        text : string
        fqid : UInt32
        room_fqid : UInt32
        text_fqid : UInt32
        fullscreen : UInt32
        hq : UInt32
        music : UInt32
        level_group : string
     }
    type [<CLIMutable>] T = {text : string}
    type [<CLIMutable>] T1 = {text : string[]}
    
    [<CLIMutable>]
    type SLabel = 
        {
            session_id : string
            correct : single
            session_id_num : single
            question : int
        }

    let saveBin file (dv:IDataView) =
        use str = System.IO.File.Create file
        ctx.Data.SaveAsBinary(dv,str,keepHidden=false)

    let saveFeatures() = 
        let colInfr = ctx.Auto().InferColumns(trainData,labelColumnName="index",groupColumns=false)
        ColInfo.setAsText ["text"] colInfr.ColumnInformation
        ColInfo.ignore ["session_id"; "elapsed_time";"index"] colInfr.ColumnInformation
        ColInfo.setAsCategorical ["page"; "level"; "hq"; "fullscreen"; "music"] colInfr.ColumnInformation
        ColInfo.showCols colInfr.ColumnInformation
        let ldr = ctx.Data.CreateTextLoader(colInfr.TextLoaderOptions)
        let dv = ldr.Load(trainData)
        let catCols = colInfr.ColumnInformation.CategoricalColumnNames |> Seq.map (fun c -> InputOutputColumnPair(c,c)) |> Seq.toArray
        let txOneHot = ctx.Transforms.Conversion.MapValueToKey(catCols)
        let txKeyToValue = ctx.Transforms.Conversion.MapKeyToValue("level_group")
        let txAll = txOneHot <!> txKeyToValue
        let dv1 = txAll.Fit(dv).Transform(dv)
        dv1.Schema |> Schema.printSchema
        //Schema.genType "SEvent" dv1.Schema


        let rs = ctx.Data.CreateEnumerable<SEvent>(dv1,false,ignoreMissingColumns=true) 
        let rsCount = PSeq.length rs
        printfn $"""Row count: {rsCount:n0}"""

        let plotSessionLengths() = 
            let ssCount = rs |> PSeq.countBy (fun x->x.session_id) |> PSeq.map snd |> PSeq.toArray
            ssCount
            |> Chart.Histogram |> Chart.withTitle $"Session length distribution; Sessions Count:{ssCount.Length}" 
            |> Chart.show
            let lgCount = rs |> PSeq.countBy (fun x->x.session_id,x.level_group) |> PSeq.map snd |> PSeq.toArray
            lgCount
            |> Chart.Histogram |> Chart.withTitle $"Session-Level_group length distribution<br>Sessions-Level_group Count:{lgCount.Length}" 
            |> Chart.show

        (*
        plotSessionLengths()
        *)


        let totalRows = rs |> PSeq.map(fun x -> x.session_id) |> PSeq.distinct |> PSeq.length
        printfn $"Total rows: {totalRows}"

        let rsg = 
            rs 
            |> PSeq.groupBy(fun x -> x.session_id,x.level_group)
            |> PSeq.map(fun ((sess,lg),xs) -> 
                let dv0 = ctx.Data.LoadFromEnumerable(xs,schemaDefinition=Schema.cleanSchema typeof<T>)
                let txw1 = ctx.Transforms.Text.NormalizeText("text")       
                let txw2 = ctx.Transforms.Text.TokenizeIntoWords("text",separators=[|','; '.'; ' '; ';'; '?'; '!'; '\''|])
                let txw3 = ctx.Transforms.Text.RemoveDefaultStopWords("text")
                let txw = txw1 <!> txw2 <!> txw3
                let dv01 = txw.Fit(dv0).Transform(dv0)
                //Schema.printSchema dv01.Schema
                let ms2 =  ctx.Data.CreateEnumerable<T1>(dv01,false,ignoreMissingColumns=true) |> Seq.collect(fun x -> if x.text <> null then x.text else [||]) |> Seq.distinct |> Seq.toArray
                let ms3 = String.Join(" ",ms2)
                let xs = Seq.cache xs
                {
                    session_id = sess 
                    level_group = lg
                    count = xs  |> Seq.length |> float32
                    elapsed_time_mean = xs |> Seq.averageBy (fun x->x.elapsed_time)
                    elapsed_tiem_std  = xs |> Seq.map(fun x-> x.elapsed_time ) |> Seq.toArray |> Statistics.ArrayStatistics.StandardDeviation |> float32
                    event_name_count = xs |> Seq.map(fun x->x.event_name) |> Seq.distinct |> Seq.length |> float32
                    name_count = xs |> Seq.map(fun x->x.name) |> Seq.distinct |> Seq.length |> float32
                    //level_count = xs |> Seq.map(fun x->x.level) |> Seq.distinct |> Seq.length |> float32
                    //page_count =xs |> Seq.map(fun x->x.page) |> Seq.distinct |> Seq.length |> float32
                    room_coor_x_mean = xs |> Seq.averageBy (fun x->x.room_coor_x)
                    room_coor_x_std = xs |> Seq.map(fun x-> x.room_coor_x ) |> Seq.toArray |> Statistics.ArrayStatistics.StandardDeviation |> float32
                    room_coor_y_mean = xs |> Seq.averageBy (fun x->x.room_coor_y)
                    room_coor_y_std = xs |> Seq.map(fun x-> x.room_coor_y ) |> Seq.toArray |> Statistics.ArrayStatistics.StandardDeviation |> float32
                    screen_coor_x_mean = xs |> Seq.averageBy (fun x->x.screen_coor_x)
                    screen_coor_x_std= xs |> Seq.map(fun x-> x.screen_coor_x ) |> Seq.toArray |> Statistics.ArrayStatistics.StandardDeviation |> float32
                    screen_coor_y_mean = xs |> Seq.averageBy (fun x->x.screen_coor_y)
                    screen_coor_y_std = xs |> Seq.map(fun x-> x.screen_coor_y ) |> Seq.toArray |> Statistics.ArrayStatistics.StandardDeviation |> float32
                    hover_duration_mean = xs |> Seq.averageBy (fun x->x.hover_duration)
                    hover_duration_std = xs |> Seq.map(fun x-> x.hover_duration ) |> Seq.toArray |> Statistics.ArrayStatistics.StandardDeviation |> float32
                    text = ms3 // (StringBuilder(),xs) ||> Seq.fold (fun acc x -> acc.Append(x)) |> fun x -> x.ToString()
                    fqid_count = xs |> Seq.map(fun x->x.fqid) |> Seq.distinct |> Seq.length |> float32
                    room_fqid_count = xs |> Seq.map(fun x->x.room_fqid) |> Seq.distinct |> Seq.length |> float32
                    text_fqid_count =  xs |> Seq.map(fun x->x.text_fqid) |> Seq.distinct |> Seq.length |> float32
                    fullscreen_count =  xs |> Seq.map(fun x-> float32 x.fullscreen) |> Seq.average
                    hq_count = xs |> Seq.map(fun x-> float32 x.hq) |> Seq.average
                    music_count = xs |> Seq.map(fun x-> float32 x.music) |> Seq.average
                    answers = [||] // to be filled later
                }        
            )

        let rsgLen = PSeq.length rsg

        let dv2 = ctx.Data.LoadFromEnumerable(rsg,schemaDefinition=Schema.cleanSchema typeof<SEventAgg>)
        Schema.printSchema dv2.Schema
        let nCols = dv2.Schema |> Seq.filter(fun x -> x.Type = NumberDataViewType.Single) |> Seq.map(fun x->x.Name) |> Seq.toArray
        let nText = dv2.Schema |> Seq.filter(fun x -> x.Type = TextDataViewType.Instance) |> Seq.map(fun x -> x.Name) |> Seq.toArray
        let ignoreCols = ["session_id"; "level_group"]

        let ciLbls = ctx.Auto().InferColumns(trainLables,"session_id")
        ColInfo.showCols ciLbls.ColumnInformation
        let dv10Ldr = ctx.Data.CreateTextLoader(ciLbls.TextLoaderOptions)
        let dvl0 = dv10Ldr.Load(trainLables)
        dvl0.Schema |> Schema.printSchema


        let answersMap = 
            ctx.Data.CreateEnumerable<SLabel>(dvl0,false,ignoreMissingColumns=true) 
            |> Seq.map(fun x -> let ts = x.session_id.Split("_q") in {x with session_id_num = float32 ts.[0]; question = int ts.[1]})
            |> Seq.groupBy (fun x -> x.session_id_num )
            |> Seq.map(fun (k,xs) -> k,xs |> Seq.map(fun y -> y.question,y.correct) |> Map.ofSeq |> Map.toSeq |> Seq.map snd |> Seq.toArray) 
            |> Seq.map(fun (i,answ) -> if answ.Length <> 18 then failwith "not all answers present" else (); (i,answ))
            |> dict

        //let level_groups = rsg  |> PSeq.map(fun x -> x.level_group) |> PSeq.distinct |> PSeq.toArray
        //let r1 = rsg |> PSeq.filter(fun x -> x.level_group = "13-22") |> PSeq.length
        //let r2 = rsg |> PSeq.filter(fun x -> x.level_group = "5-12") |> PSeq.length
        //let r3 = rsg |> PSeq.filter(fun x -> x.level_group = "0-4") |> PSeq.length

        let rsgAns = rsg |> PSeq.map(fun x -> {x with answers = answersMap.[x.session_id]}) |> PSeq.toArray
        rsgAns.Length

        let dvAns = ctx.Data.LoadFromEnumerable(rsgAns,schemaDefinition=Schema.cleanSchema typeof<SEventAgg>)
        saveBin dvFile dvAns

(*
Featurize.saveFeatures()
*)

let dvAns2 = ctx.Data.LoadFromBinary(dvFile)
Schema.printSchema dvAns2.Schema

let seBinClassification = ctx.Auto().BinaryClassification(labelColumnName="target",featureColumnName="Features",useFastTree=true,useLgbm=true)

let ftrCols =
    //dvAns.Schema |> Seq.map(fun x->x.Name) |> Seq.toArray
    [| "count"; "elapsed_time_mean";
        "elapsed_tiem_std"; "event_name_count"; "name_count"; "room_coor_x_mean";
        "room_coor_x_std"; "room_coor_y_mean"; "room_coor_y_std";
        "screen_coor_x_mean"; "screen_coor_x_std"; "screen_coor_y_mean";
        "screen_coor_y_std"; "hover_duration_mean"; "hover_duration_std"; "text";
        "fqid_count"; "room_fqid_count"; "text_fqid_count"; "fullscreen_count";
        "hq_count"; "music_count";|]

let seBase =    
    let fac (ctx:MLContext) p = 
        ctx.Transforms.Concatenate("Features",ftrCols) |> asEstimator
    SweepableEstimator(fac,new SearchSpace())

let grammar = 
    [
        Alt [
                Estimator (E.seTextFeaturize "text")
                Estimator (E.seTextHashedNGrams "text")
        ]            
        Estimator seBase       
        Opt(
            Alt [
                Estimator (E.seFtrSelCount 10)
                Estimator (E.seFtrSelMutualInf "target")
            ])
        Opt(
            Alt [
                Alt ([(1,10); (11,20); (21,30); (31,100)] |> List.map(E.seNorm>>Estimator))
                Estimator E.seNormLpNorm
                Estimator E.seNormLogMeanVar
                Estimator E.seNormMeanVar
                Alt([0.1f .. 0.5f .. 4.0f] |> List.pairwise |> List.map(fun (a,b) -> a, b - 0.001f)  |> List.map(E.seGlobalContrast>>Estimator))
                Estimator E.seNormMinMax
                //Estimator E.seNormRobustScaling
                Estimator (E.seNormSupBin "target")
            ])
        Opt (Estimator E.seWhiten)
        Opt (
            Alt [
                Estimator (E.seProjPca (2,10))
                Estimator (E.seKernelMap (2,10))
            ])
        Pipeline seBinClassification
    ]

type TTarget() = 
    [<DefaultValue>] val mutable target:bool

type TFilter() =
    [<DefaultValue>] val mutable answers:single[]
    [<DefaultValue>] val mutable level_group : string

let setTarget answ (tIn:TFilter) (tOut:TTarget)  =  tOut.target <- tIn.answers.[answ] > 0.f

let expFac question dv timeout (p:SweepablePipeline) =
    ctx.Auto()
        .CreateExperiment()
        .SetBinaryClassificationMetric(BinaryClassificationMetric.AreaUnderPrecisionRecallCurve,"target")
        .SetDataset(dv,3)
        .SetTrainingTimeInSeconds(timeout)        
        .SetPipeline(p)
        .SetMonitor(
            let s : TrialSettings ref = ref Unchecked.defaultof<_>
            let printLine (isDone:bool) (r:TrialResult) =  printfn $"""{question} M: {r.Metric} {isDone} - {s.Value.Parameter.["_pipeline_"]}"""
            {new IMonitor with
                 member this.ReportBestTrial(result) = printLine true result 
                 member this.ReportCompletedTrial(result) = printLine false result 
                 member this.ReportFailTrial(settings, ``exception``) = printfn "%A" ``exception``; printfn $"%A{settings}"
                 member this.ReportRunningTrial(setting) = s.Value <- setting            
            })


let train lvlGrpu answ =
    let dv2 = ctx.Data.FilterByCustomPredicate(dvAns2,fun (t:TFilter) -> t.level_group <> lvlGrpu)
    let txTgt = ctx.Transforms.CustomMapping(setTarget answ,contractName=null)
    let dv3 = txTgt.Fit(dv2).Transform(dv2)
    Schema.printSchema dv3.Schema 
    let oPl,oAcc,rslt = Optimize.run CA.OptimizationKind.Maximize (expFac answ dv3 600u) grammar
    let mdlPath = root @@ $"model_{answ}.bin"
    ctx.Model.Save(rslt.Model,dv3.Schema,mdlPath)
    rslt,oPl

//train "0-4" 2


let lvlGrpAns = 
    [
        "0-4", [0..3]
        "5-12", [4..11]
        "13-22",[12..17]
    ]
    |> List.collect(fun (l,xs) -> xs |> List.map (fun y -> l,y))

let rslts = lvlGrpAns |> List.map(fun (l,a) -> async{return train l a;}) |> Async.Parallel |> Async.RunSynchronously








