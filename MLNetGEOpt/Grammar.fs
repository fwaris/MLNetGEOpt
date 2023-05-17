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
        //let i = if i >= genome.Length then 0 else i  //no wrap around in our case
        match t with
        | Pipeline _ | Estimator _ -> (t::acc),i 
        | Opt t'  -> 
            if genome.[i] = 0 then 
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

    let translate genome grammar =
        (([],0),grammar) ||> List.fold (fun (acc,i) t -> tranlateTerm genome (acc,i) t)

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
        (*
            What is a good estimate of genome size?
            Genome 'size' has two aspects:
            A) the length of the genome - or the number of integer slots
            B) the range of each integer slot
            The grammar has only 2 type of decision points Opt and Alt terms.
            The rest are deterministic - no choice.
            The choice for Opt requires only a boolean value (0 or 1)
            The choice for Alt requires a value with range 0..(N-1) where N is the number of child terms in the Alt term.
            The largest possible Alt term can be used to set the range of all integer slots.
            Under GE, a genome can be reused by 'wrapping around' if the end of the gnome is reached 
            and there are still grammar terms remaining.
            However, here, we can apriori determine the length of the longest 'sentence' for a given grammar. 
            This is because we don't have recursion. Nesting is allowed though, i.e. Opt and Alt terms may contain child Opt and Alt terms (along with other terms).
            Genome length is therefore the longest possible path in the nested grammar.
            *Note: We can do slightly better here.* 
            Because we know the maximum length, we don't have to wrap around.
            This allows us to specify further restrictions on the ranges of individual integer slots.
            We can perform a breadth first search and for each level we can take the maximum range
            as the allowed range for the corresponding integer slot. 
            So, instead of setting all integer slots to the same range, we can customize
            individual slots to be more restrictive - reducing the volume of the search space.
        *)
        let choiceRange xs = 
            if List.length xs = 0 then 
                0 
            else  
                xs |> List.map (function 
                | Alt xs -> xs.Length - 1
                | Opt _ -> 1 
                | _ -> 0) 
                |> List.max
        let rec loop acc nodes =
            let next = nodes |> List.collect (function Alt xs -> xs | Opt x -> [x] | _ -> [])
            if List.isEmpty next then
                List.rev acc
            else
                let nextRange = choiceRange next
                let acc = nextRange::acc
                loop acc next
        loop [choiceRange g] g |> List.filter (fun x -> x > 0)

