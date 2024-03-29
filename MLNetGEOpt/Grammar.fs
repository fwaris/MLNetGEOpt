namespace MLNetGEOpt
open Microsoft.ML.AutoML

//meta grammar to describe the grammar for the grammatical evolution process

type Term = 
    | Opt of Term 
    | Pipeline of (unit -> SweepablePipeline)
    | PipelineN of string* (unit -> SweepablePipeline)
    | Estimator of (unit -> SweepableEstimator)
    | Alt of Term list
    | Union of Term list

type Grammar = Term list
    
[<RequireQualifiedAccess>]
module Grammar =
    let rec private getId term =
        match term with 
        | Pipeline _ -> "Pipeline"
        | PipelineN (n,_) -> $"Pipeline {n}"
        | Estimator e -> 
            let e = e() 
            let t = match e.SearchSpace.TryGetValue Search.TERM_ID with 
                    | true,p -> p :?> Microsoft.ML.SearchSpace.Option.ChoiceOption
                    | _      -> failwith "Estimator search term must have a paramater named Search.TERM_ID. See Search.withId function"
            t.Choices.[0].AsType<string>()
        | _ -> "unexpected"        

    let rec private tranlateTerm (genome:int[]) (acc,i,tcount) (t:Term) =
        let i = if i >= genome.Length then 0 else i  //wrap around
        match t with
        | Pipeline _ | PipelineN _ | Estimator _ -> (t::acc),i,tcount
        | Opt t'  -> 
            if genome.[i] % 2 = 0 then 
                tranlateTerm genome (acc,i+1,tcount+1) t'
            else
                acc,(i+1),tcount+1
        | Alt ts ->            
            if ts.Length = 0 then failwith "Alt term must have atleast 1 child term"
            if ts.Length = 1 then 
                //Alt has only 1 term - there is no choice here
                tranlateTerm genome (acc,i,tcount) ts.Head
            else 
                let t' = ts.[genome.[i] % ts.Length]     
                tranlateTerm genome (acc,i+1,tcount+1) t'
        | Union ts -> 
            match ts with 
            | [] -> acc,i,tcount
            | t::rest -> 
                let acc,i,tcount = tranlateTerm genome (acc,i,tcount) t
                tranlateTerm genome (acc,i,tcount) (Union rest)

    let translate grammar genome  =
        let terminals,i,len =
            (([],0,0),grammar) 
            ||> List.fold (fun (acc,i,tcount) t -> tranlateTerm genome (acc,i,tcount) t)
        List.rev terminals, len
        

    let toPipeline terminals = 
        let h,ts =
            match terminals with
            | Pipeline p::rest                        -> p(),rest
            | PipelineN (n,p)::rest                   -> p(),rest
            | (Estimator e1)::(Estimator e2)::rest    -> (e1().Append(e2())),rest
            | (Estimator e1)::(Pipeline e2)::rest  
            | (Estimator e1)::(PipelineN(_,e2))::rest -> (e1().Append(e2())),rest
            | _                                       -> failwith "Given list should be only Pipeline or Estimator terms with atleast two estimators or one pipeline"
        (h,ts) 
        ||> List.fold (fun acc t ->
            match t with 
            | Estimator e -> acc.Append(e())
            | Pipeline p 
            | PipelineN (_,p) -> (acc,p().Estimators) ||> Seq.fold(fun acc kv -> acc.Append(kv.Value))
            | _          -> failwith "Only Pipeline or Estimator terms expected. Ensure 'translate' is called")

    let pipelineHash ctx terminals =
        let mutable count = 0
        let sb = System.Text.StringBuilder()
        let (!+) (a:string) = sb.AppendLine(a) |> ignore
        terminals 
        |> List.iter (fun t -> 
            let id = getId t
            !+ id)
        sb.ToString()

    let printPipeline ctx terminals = printfn "%A" (pipelineHash ctx terminals)

    let estimateGenomeSize (g:Grammar) =
        let rec skipHead xs = //skip no-choice terminals in the beginning from search space
            match xs with
            | [] -> []
            | Pipeline _::rest | Estimator _ :: rest -> skipHead  rest
            | xs -> xs 
        let g' = skipHead g
        let rec maxChoices d xs =
            match xs with 
            | [] -> d
            | Alt xs :: rest -> 
                let d' = xs |> List.map (fun t -> maxChoices 0 [t]) |> List.max
                let d'' = d + d' + 1
                maxChoices d'' rest
            | Opt t :: rest ->
                let d' = maxChoices 0 [t]
                let d'' = d + d' + 1
                maxChoices d'' rest
            | Union xs :: rest ->
                let d' = xs |> List.sumBy (fun t -> maxChoices 0 [t])
                let d'' = d + d'
                maxChoices d'' rest
            | Pipeline _::rest -> maxChoices d rest
            | PipelineN _ ::rest -> maxChoices d rest
            | Estimator _::rest -> maxChoices d rest

        maxChoices 0 g'
