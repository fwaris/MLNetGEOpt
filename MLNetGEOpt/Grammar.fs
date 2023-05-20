namespace MLNetGEOpt
open Microsoft.ML.AutoML

//meta grammar to describe the grammar for the grammatical evolution process

type Term = 
    | Opt of Term 
    | Pipeline of SweepablePipeline 
    | Estimator of SweepableEstimator  
    | Alt of Term list
    | Union of Term list

type Grammar = Term list
    
[<RequireQualifiedAccess>]
module Grammar =

    let rec private tranlateTerm (genome:int[]) (acc,i) (t:Term) =
        let i = if i >= genome.Length then 0 else i  //wrap around
        match t with
        | Pipeline _ | Estimator _ -> (t::acc),i 
        | Opt t'  -> 
            if genome.[i] % 2 = 0 then 
                tranlateTerm genome (acc,i+1) t'
            else
                acc,(i+1)
        | Alt ts ->            
            if ts.Length = 0 then failwith "Alt term must have atleast 1 child term"
            if ts.Length = 1 then 
                //Alt has only 1 term - there is no choice here
                tranlateTerm genome (acc,i) ts.Head
            else 
                let t' = ts.[genome.[i] % ts.Length]     
                tranlateTerm genome (acc,i+1) t'
        | Union ts -> 
            match ts with 
            | [] -> acc,i
            | t::rest -> 
                let acc,i = tranlateTerm genome (acc,i) t
                tranlateTerm genome (acc,i+1) (Union rest)

    let translate grammar genome  =
        let terminals,len =
            (([],0),grammar) 
            ||> List.fold (fun (acc,i) t -> tranlateTerm genome (acc,i) t)
        List.rev terminals, len
        

    let toPipeline terminals = 
        let h,ts =
            match terminals with
            | Pipeline p::rest -> p,rest
            | (Estimator e1)::(Estimator e2)::rest -> e1.Append(e2),rest
            | (Estimator e1)::(Pipeline e2)::rest  -> e1.Append(e2),rest
            | _                                    -> failwith "Given list should be only Pipeline or Estimator terms with atleast two estimators or one pipeline"
        (h,ts) 
        ||> List.fold (fun acc t ->
            match t with 
            | Estimator e -> acc.Append(e)
            | Pipeline p -> (acc,p.Estimators) ||> Seq.fold(fun acc kv -> acc.Append(kv.Value))
            | _          -> failwith "Only Pipeline or Estimator terms expected. Ensure 'translate' is called")

    let esimateGenomeSize (g:Grammar) =
        let rec skipHead xs = //skip no-choice terminals in the beginning from search space
            match xs with
            | [] -> []
            | Pipeline _::rest | Estimator _ :: rest -> skipHead  rest
            | xs -> xs 

        let g' = skipHead g
        let rec maxDepth d xs =
            match xs with 
            | [] -> d
            | Alt xs :: rest -> 
                let d' = maxDepth (d+1) xs
                let d'' = maxDepth d rest 
                max d' d''
            | Opt t :: rest ->
                let d' = maxDepth (d+1) [t]
                let d'' = maxDepth d rest
                max d' d''
            | Union xs :: rest ->
                let d' = maxDepth d xs
                let d'' = maxDepth d rest
                max d' d''
            | Pipeline _::rest -> maxDepth d rest
            | Estimator _::rest -> maxDepth d rest

        maxDepth 0 g'





