namespace MLNetGEOpt
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.AutoML
open Microsoft.ML.SearchSpace
open MLUtils.Pipeline

//meta grammar to describe the grammar for the grammatical evolution process
//
(*
What do we need?
Given:
- linear genome composed of float values
- genome (genotype) translated to AST (phenotype)
- AST evaluated for fitness
- population of linear genome evolved via 'search engine'
Grammar:
- start, non-terminals, terminals [keywords | [value (domain / range)]], production rules
*)

type Term = 
    | Opt of Term 
    | Pipeline of SweepablePipeline 
    | Estimator of SweepableEstimator  
    | Alt of Term list
    | Union of Term list

type Grammar = Term list

type Srch =
    static member Double(s:string,lo:float,hi:float,?isLog) = let ss = SearchSpace() in ss.Add(s,Option.UniformDoubleOption(lo,hi,?logBase=isLog)); ss

type Fac1 = 
    static member  fac1 (ctx:MLContext) : IEstimator<ITransformer> = ctx.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation("a",labelColumnName="b") :> IEstimator<ITransformer>
    
[<RequireQualifiedAccess>]
module T =

    let rec tranlateTerm (genome:int[]) (acc,i) (t:Term) =
        let i = if i >= genome.Length then 0 else i
        match t with
        | Pipeline _ | Estimator _ -> (t::acc),i 
        | Opt t'  -> 
            if genome.[i] = 0 then 
                tranlateTerm genome (acc,i+1) t'
            else
                acc,(i+1)
        | Alt ts ->
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

    let toPipeline ts = 
        let h,ts =
            match ts with
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
            However, here we can determine the length of the longest 'sentence' for a given grammar. 
            This is because we don't have recursion. Nesting is allowed though, i.e. Opt and Alt terms may contain child Opt and Alt terms (along with other terms).
            Genome length is the longest possible path in the nested grammar.
            *However, we can do slightly better here.* Because we know the maximum length, we don't have to wrap around.
            This allows us to specify further restrictions on the ranges of individual integer slots.
            We can perform a breadth first search and for each level we can take the maximum range
            as the allowed range for the corresponding integer slot. 
            So instead of setting all integer slots to the same range, we can customize
            individual slots to be more restrictive - reducing the volume of the search space.

        *)
        ()