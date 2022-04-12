"""Provides functions to load entire benchmark result datasets
"""
import os
import io
import glob
import gzip
import tarfile
import warnings

import numpy
import pandas

from .parse import IorOutput, MdWorkbenchOutput, ElbenchoCsvOutput
from .contention import validate_contention_dataset, JobOverlapError, ShortJobError

def _load_ior_output_stream(stream, fname, all_results=None):
    """Recursive function that loads one or more IOR output files

    Args:
        stream (io.TextIOWrapper): file-like object containing the stdout of
            an IOR job or jobs.
        fname (str): file name associated with stream.
        all_results (pandas.DataFrame or None): Dataframe to which loaded
            results should be appended.

    Returns:
        pandas.DataFrame: all_results with newly loaded data appended
        as new rows.
    """
    if isinstance(stream, tarfile.TarFile):
        for member in stream.getmembers():
            handle = stream.extractfile(member)
            if handle: # directories will have handle = None
                all_results = _load_ior_output_stream(
                    io.TextIOWrapper(handle),
                    member.name,
                    all_results)
    else:
        result = IorOutput(stream, normalize_results=True)
        if not result or 'results' not in result:
            warnings.warn('invalid output in {}'.format(fname))
            return all_results

        result.add_filename_metadata(fname)
        results_df = pandas.DataFrame.from_dict(result['results']).dropna(subset=['bw(mib/s)'])
        results_df['filename'] = fname

        # graft in some columns from summary lines - indices should be the same
        summaries_df = pandas.DataFrame.from_dict(result['summaries'])
        if 'aggs(mib)' in summaries_df:
            if 'stonewall_bytes_moved' in results_df:
                na_indices = results_df[results_df['stonewall_bytes_moved'].isna()].index
                if na_indices.shape[0] > 0:
                    results_df.loc[na_indices, 'stonewall_bytes_moved'] = summaries_df.loc[na_indices, 'aggs(mib)'] * 2**20
            else:
                results_df['stonewall_bytes_moved'] = summaries_df['aggs(mib)'] * 2**20

        if all_results is None:
            all_results = results_df
        else:
            if len(all_results.columns) != len(results_df.columns):
                warn_str = 'inconsistent input file: {}' + \
                           ' (file only has {:d} of {:d} expected columns)\n' +\
                           'this file: {}\n' + \
                           'expected:  {}\n' + \
                           'diff:      {}'
                warnings.warn(warn_str.format(
                                fname,
                                len(results_df.columns),
                                len(all_results.columns),
                                ','.join(results_df.columns),
                                ','.join(all_results.columns),
                                ','.join(list(set(all_results.columns) ^ set(results_df.columns)))))
            all_results = pandas.concat((all_results, results_df))

    return all_results

def load_ior_output_files(input_glob):
    """Finds and loads one or more IOR output files.

    Args:
        input_glob (str): A path or glob to one or more IOR output files.  Such
            files may be ASCII files, gzipped ASCII, or tar files containing
            multiple IOR output ASCII files.

    Returns:
        pandas.DataFrame: benchmark results from the files matching input_glob
    """
    all_results = None
    if isinstance(input_glob, str):
        input_globs = [input_glob]
    else:
        input_globs = input_glob

    for input_glob in input_globs:
        for output_f in glob.glob(input_glob):
            if output_f.endswith('.tar') or output_f.endswith('.tgz'):
                stream = tarfile.open(output_f, 'r')
            elif output_f.endswith('.gz'):
                stream = gzip.open(output_f, 'r')
            else:
                stream = open(output_f, 'r')
            all_results = _load_ior_output_stream(stream, output_f, all_results)

    if all_results is None:
        raise ValueError(f"Non-existent dataset {input_glob}")
    all_results = all_results.reset_index(drop=True)
    all_results['nproc'] = all_results['nodes'] * all_results['ppn']
    if all_results is None:
        warnings.warn(f'Found no valid results in {input_glob}!')
    else:
        print('Found {:d} results in {}.'.format(
            all_results.shape[0], ", ".join(input_globs)))
    return all_results

def load_ior_vs_setsize_results(input_glob, filter_setsizes_below_gibs=65):
    """Finds and loads IOR output files for performance-vs-setsize analysis.

    Args:
        input_glob (str): A path or glob to one or more IOR output files.  Such
            files may be ASCII files, gzipped ASCII, or tar files containing
            multiple IOR output ASCII files.
        filter_setsizes_below_gibs (int): Exclude measurements that had a
            setsize smaller than this value (in GiBs)

    Returns:
        pandas.DataFrame: benchmark results from the files matching input_glob
    """
    results = None
    for output_f in glob.glob(input_glob):
        if output_f.endswith("gz"):
            records = IorOutput(gzip.open(output_f, 'r'), normalize_results=True)
        else:
            records = IorOutput(open(output_f, 'r'), normalize_results=True)

        frame = pandas.DataFrame.from_records(records['results'])
        if results is None:
            results = frame
        else:
            results = pandas.concat((results, frame), ignore_index=True)
    results['gib_moved'] = results['bw(mib/s)'] * results['total(s)'] / 1024.0
    filt = (results['access'] == 'read') | (results['access'] == 'write')
    filt &= results['bw(mib/s)'] > 0.0

    results = results[filt].sort_values('timestamp').reset_index(drop=True).copy()
    results['timestamp'] = results['timestamp'].apply(int)
    tmp = results['gib_moved'].values
    tmp[1::2] = results['gib_moved'].iloc[::2]

    results['setsize_gib'] = tmp
    results["setsize_gib_int"] = results['setsize_gib'].apply(numpy.rint).astype(numpy.int32)
    filt = results['setsize_gib_int'] >= filter_setsizes_below_gibs

    print('Found {:d} runs ({:d} results) in {}.'.format(
        results.groupby("setsize_gib_int").count().iloc[0, 0] // 2, # /2 because 1 run = write+read
        results.shape[0],
        input_glob))


    return results[filt].copy()

def load_contention_dataset(dataset_glob, dataset_id=None, as_records=False):
    """Attempts to load a benchmark output file.

    Tries to instantiate the classes in n10storage.parse using the given output
    files until it finds one that doesn't error out.  It then takes the first
    'record' of the resulting dictionary and assumes the following keys exist:

    - access (read or write)
    - ordering (random or otherwise)
    - bw(mib/s) - for bandwidth tests
    - iops - for iops or metadata tests

    It also assumes that the 'header' key of the resulting dictionary contains

    - start
    """
    records = []
    for filename in glob.glob(dataset_glob):
        record = None
        for loader in IorOutput, MdWorkbenchOutput, ElbenchoCsvOutput:
            try:
                record = loader(open(filename, "r"), normalize_results=True)['results'][0]
                break
            except KeyError:
                pass
        if record is None:
            warnings.warn("{} does not contain valid output".format(os.path.basename(filename)))
            continue
        basename = os.path.basename(filename)

        # decode job metadata from filename - new way (secondary_quiet.7p-1s.2125435.out)
        if basename.startswith("primary") or basename.startswith("secondary"):
            access = record.get("access")
            metric = "bw"
            if loader == MdWorkbenchOutput:
                metric = "metadata"
                access = "both"
            elif record['ordering'] == 'random':
                metric = "iops"
            workloadid_contention, nodect, dataset_id, _ = basename.split('.', 3)
            workload_id, contention = workloadid_contention.split("_")
            primary_nodes = int(nodect.split("p", 1)[0])
            secondary_nodes = int(nodect.split("-", 1)[-1].split("s", 1)[0])
            record['dataset_id'] = dataset_id
            record['workload_id'] = workload_id
        else:
            # decode job metadata from filename
            access_metric_contention, nodect, _ = basename.split('.', 2)
            access, metric, contention = access_metric_contention.split("_")
            primary_nodes = int(nodect.split("b", 1)[0])
            secondary_nodes = int(nodect.split("-", 1)[-1].split("i", 1)[0])

        # add job metadata to record
        record.update({
            "primary_nodes": primary_nodes,
            "secondary_nodes": secondary_nodes,
            "access": access,
            "metric": metric,
            "contention": contention,
            "workload": "{} {}".format(access, metric),
            "filename": basename,
        })
        if dataset_id:
            record.update({"dataset_id": dataset_id})

        if metric == "bw":
            record["performance"] = record["bw(mib/s)"]
        elif metric == "iops":
            record["performance"] = record["iops"]
        elif metric == "metadata":
            record["performance"] = record["iops"]
        else:
            raise ValueError(f"unknown metric {metric}")

        records.append(record)

    # set the primary workload - always the first to run during the quiet tests
    # TODO: think about this - is it correct?  shouldn't we ensure that
    # primary_nodes and primary_workload are always consistent?  as-written,
    # the definition used here is dependent on the nature of the ordering
    # within the slurm script used to generate the dataset
    min_starts = {}
    for record in records:
        if record["contention"] != "quiet":
            continue
        dataset_id = record["dataset_id"]

        if dataset_id not in min_starts:
            min_starts[dataset_id] = {}
        rec = min_starts[dataset_id]

        if "min start" not in rec or rec["min start"] > record["start"]:
            rec.update({
                "min start": record["start"],
                "primary workload": record["workload"],
            })

    for record in records:
        record["primary_workload"] = min_starts\
            .get(record.get("dataset_id"), {})\
            .get("primary workload")

        if 'workload_id' not in record:
            if record["primary_workload"] == record["workload"]:
                record["workload_id"] = "primary"
            else:
                record["workload_id"] = "secondary"

    if not records:
        raise ValueError("Invalid datasets")

    if as_records:
        return records

    return pandas.DataFrame.from_records(records)

def load_contention_datasets(dataset_glob_map, use_cache=True, validate=True):
    """Loads contention datasets

    Args:
        dataset_glob_map (dict): Keyed by a path glob that contains exactly
            one {} which will be substituted for dataset ids.  Values should
            be lists of strings, each containing a dataset id which will be
            substituted within the key to resolve a set of matching IOR input
            files.
        use_cache (bool): Attempt to load and/or save the results to an
            intermediate cache file.

    Returns:
        pandas.DataFrame: benchmark results from the files matching input_glob
    """
    dataframe = None
    new_datasets = 0
    for dataset_glob, dataset_ids in dataset_glob_map.items():
        if dataset_glob.startswith("_"):
            continue
        cache_file = None
        if use_cache:
            filepath = os.path.dirname(dataset_glob)
            if '*' in filepath:
                warnings.warn(f"* found in {filename}; not using cache")
            else:
                cache_file = os.path.join(filepath, "dataset_summary.csv")

        if cache_file and os.path.isfile(cache_file):
            dataframe = pandas.read_csv(cache_file)
            print(f"Loaded dataset from {cache_file}")
        else:
            for dataset_id in dataset_ids:
                new_datasets += 1
                subframe = load_contention_dataset(
                    dataset_glob.format(dataset_id),
                    dataset_id)
                if dataframe is None:
                    dataframe = subframe
                else:
                    dataframe = pandas.concat((dataframe, subframe))

        dataframe = dataframe.reset_index()
        if cache_file and new_datasets:
            dataframe.to_csv(cache_file)
            print(f"Saved dataset to {cache_file}")

    # pandas tries to turn dataset_ids into ints - don't do that
    dataframe['dataset_id'] = dataframe['dataset_id'].astype(str)

    if validate:
        try:
            validate_contention_dataset(dataframe)
        except (JobOverlapError, ShortJobError) as err:
            warnings.warn("Caught {}".format(err))
            pass
    return dataframe
