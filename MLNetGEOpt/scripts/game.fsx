#load "packages.fsx"
open System
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.AutoML
open Microsoft.ML.SearchSpace
open MLNetGEOpt
open MLUtils
open MLUtils.Pipeline
open MathNet.Numerics
open System.Text
open FSharp.Collections.ParallelSeq

let trainData = @"E:\s\kaggle\predict-student-performance-from-game-play\train.csv"
let trainLables = @"E:\s\kaggle\predict-student-performance-from-game-play\train_labels.csv"

let ctx = MLContext()
let colInfr = ctx.Auto().InferColumns(trainData,labelColumnName="index",groupColumns=false)
ColInfo.setAsText ["text"] colInfr.ColumnInformation
ColInfo.ignore ["session_id"; "elapsed_time";"index"] colInfr.ColumnInformation
ColInfo.setAsCategorical ["page"; "level"; "hq"; "fullscreen"; "music"] colInfr.ColumnInformation
ColInfo.showCols colInfr.ColumnInformation
let ldr = ctx.Data.CreateTextLoader(colInfr.TextLoaderOptions)
let dv = ldr.Load(trainData)
let catCols = colInfr.ColumnInformation.CategoricalColumnNames |> Seq.map (fun c -> InputOutputColumnPair(c,c)) |> Seq.toArray
let txOneHot = ctx.Transforms.Conversion.MapValueToKey(catCols)
let dv1 = txOneHot.Fit(dv).Transform(dv)
dv1.Schema |> Schema.printSchema
//Schema.genType "SEvent" dv1.Schema

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
    level_group : UInt32
 }

let rs = ctx.Data.CreateEnumerable<SEvent>(dv1,false,ignoreMissingColumns=true) 

[<CLIMutable>]
type SEventFtr = 
    {
        session_id: Single
        level_group: uint32 // Key<UInt32, 0-2> {KeyValues}
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
    }

type [<CLIMutable>] T = {text : string}
type [<CLIMutable>] T1 = {text : string[]}

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
        {
            session_id = sess 
            level_group = lg
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
        }        
    )

let dv2 = ctx.Data.LoadFromEnumerable(rsg,schemaDefinition=Schema.cleanSchema typeof<SEventFtr>)
Schema.printSchema dv2.Schema
