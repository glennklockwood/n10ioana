import sys
import numpy
import pandas

CONTENTION_METRIC_NAMES = {
    "bw": "Bandwidth",
    "iops": "IOPS",
    "metadata": "Metadata",
}

CONTENTION_METRIC_UNITS = {
    "bw": "MiB/s",
    "iops": "IOPS",
    "metadata": "IOPS",
}

class ShortJobError(ValueError):
    pass

class JobOverlapError(ValueError):
    pass

class IncompleteDatasetError(ValueError):
    def __init__(self, message, primary_nodes, contention):
        self.primary_nodes = primary_nodes
        self.message = message
        self.contention = contention
        super().__init__(self.message)

def calculate_contention_overlap(dataframe):
    """Calculates overlap of contention jobs
    
    Creates a dataframe that is
    
    - indexed by dataset_id (slurm jobid)
    - columns are a multiindex of
        - primary_nodes
        - contention (noisy or quiet)
        - time (overlapping, non-overlapping, or total)
    
    ``time=overlapping`` is the number of seconds that primary and secondary
    jobs overlapped and is the difference between the fastest job completion
    timestamp and the slowest starting job timestamp.  Negative overlaps
    indicate that the jobs did not overlap at all, so these are set to 0.0.
    
    ``time=non-overlapping`` is the number of seconds where one job (primary)
    or secondary) is running while the other job is not.  It is the difference
    between ``time=total`` and ``time=overlapping``.  It only really makes
    sense in the noisy case
    
    ``time=total`` is the number of seconds that primary and secondary
    jobs were running in isolation and is the difference between the slowest
    job completion timestamp and the fastest job start timestamp.  It also
    contains any gap between primary and secondary phases being run in the
    quiet case.
    
    Args:
        dataframe (pandas.DataFrame): output of load_contention_datasets()
    Returns:
        pandas.DataFrame: Overlap between primary and secondary workload jobs
            in seconds
    """
    pivoted_df = dataframe.pivot_table(
        index=["dataset_id"],
        values=["start", "end"],
        columns=["primary_nodes", "contention", "workload_id"])
    pivoted_df.columns.names = ["start/end"] + pivoted_df.columns.names[1:]

    overlaps = pandas.DataFrame(
        index=pandas.Index(data=dataframe['dataset_id'].unique(), name="dataset_id"),
        columns=pandas.MultiIndex.from_product([
            numpy.sort(dataframe['primary_nodes'].unique()),
            dataframe['contention'].unique(),
            ["overlapping", "non-overlapping", "total"],
        ], names=("primary_nodes", "contention", "time")))

    for numnode in pivoted_df.columns.get_level_values("primary_nodes").unique():
        for contention in pivoted_df.columns.get_level_values("contention").unique():
            # (earliest end) - (latest start), in seconds
            try:
                overlaps.loc[:, (numnode, contention, "overlapping")] = (
                        pivoted_df['end', numnode, contention].min(axis=1)
                        - pivoted_df['start', numnode, contention].max(axis=1)
                ).apply(lambda x: max(0.0, x))
            except KeyError as error:
                raise IncompleteDatasetError(
                    f"incomplete dataset for primary_nodes={numnode} contention={contention}",
                    numnode,
                    contention)

            # (latest end) - (earliest start), in seconds
            overlaps.loc[:, (numnode, contention, "total")] = \
                pivoted_df['end', numnode, contention].max(axis=1) \
                - pivoted_df['start', numnode, contention].min(axis=1) \

            # (latest end) - (earliest start) - (overlap), in seconds
            overlaps.loc[:, (numnode, contention, "non-overlapping")] = \
                overlaps.loc[:, (numnode, contention, "total")] \
                - overlaps.loc[:, (numnode, contention, "overlapping")]

    return overlaps

def validate_contention_dataset(dataframe, min_overlap=0.80, min_walltime=45, min_overlap_warn=0.90, quiet=False):
    PRINT_COLS = [
        "dataset_id", "start", "end", "primary_nodes", "contention", "workload_id"
    ]
    
    def vprint(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)
    
    # ensure all jobs ran for minimum amount of time
    walltime = (dataframe['end'] - dataframe['start']).min()
    if walltime < min_walltime:
        idxmin = (dataframe['end'] - dataframe['start']).idxmin()
        vprint("!!! Short walltime detected:")
        vprint("!!! " + "\n!!! ".join(
            str(dataframe.loc[idxmin][PRINT_COLS]).splitlines()))
        raise ShortJobError(f"shortest walltime {walltime}s below {min_walltime}s\n")
    
    # create dataframe of start/end times for each job
    timestamps = dataframe.pivot_table(
        index=["dataset_id"],
        values=["start", "end"],
        columns=["primary_nodes", "contention", "workload_id"])
    timestamps.columns.names = ["start/end"] + timestamps.columns.names[1:]

    # create dataframe of differences between start times, end times
    sec = timestamps.loc[:, (slice(None), slice(None), slice(None), "secondary")]
    try:
        pri = timestamps.loc[:, (slice(None), slice(None), slice(None), "primary")]
    except KeyError:
        pri = None
    if pri is not None:
        deltatime = pandas.DataFrame(
            data=sec.values - pri.values,
            index=timestamps.index,
            columns=timestamps.columns.droplevel("workload_id").drop_duplicates())

    # make sure quiet jobs all start at least 45 seconds apart, irrespective
    # of dataset_id.  this ensures that two quiet jobs were never
    # overlapping.  also identifies duplicate jobs
    # create a monotonic timeline of quiet start timestamps
    quiet_starts = dataframe[dataframe['contention'] == 'quiet']\
        .sort_values("start")[PRINT_COLS].copy()
    # create a column of deltas between quiet start times
    quiet_starts["delta"] = numpy.concatenate(
        ([numpy.nan],
         quiet_starts["start"].iloc[1:].values \
         - quiet_starts["start"].iloc[:-1].values))
    if quiet_starts["delta"].min() < min_walltime:
        # find index of the smallest start time
        idxmin = numpy.nanargmin(quiet_starts["delta"].values)
        # identify the two jobs that created this minimal start time
        vprint("!!! Rapid quiet starts detected:")
        vprint("!!! " + "\n!!! ".join(
            (str(quiet_starts.iloc[idxmin - 1:idxmin + 1].T)).splitlines()))
        raise JobOverlapError(f"rapid quiet starts found")

    # calculate overlap of noisy jobs
    overlaps = calculate_contention_overlap(dataframe)

    noisy_stats = None
    for timeframe in "overlapping", "non-overlapping", "total":
        tmp = overlaps.loc[:, (slice(None), "noisy", timeframe)]
        tmp.columns = tmp.columns.droplevel("contention").droplevel("time")

        flat_series = pandas.melt(
            frame=tmp,
            col_level="primary_nodes",
            ignore_index=False
        ).reset_index().set_index(["dataset_id", "primary_nodes"]).iloc[:, 0]
        flat_series.name = timeframe

        if noisy_stats is None:
            noisy_stats = flat_series.to_frame()
        else:
            noisy_stats[timeframe] = flat_series
    noisy_stats.columns = [x + "_secs" for x in noisy_stats.columns]
    noisy_stats["overlapping_frac"] = noisy_stats["overlapping_secs"] / noisy_stats["total_secs"]

    # check noisy measurements to ensure they actually overlapped.  even if
    # two jobs overlapped for a full 60 seconds, one job taking significantly
    # longer (e.g., 75 seconds) will reflect performance of the last 20% of
    # the job running in isolation.  this will understate the actual
    # performance under contention
    if min_overlap_warn is None:
        min_overlap_warn = 1 - (1 - min_overlap) / 2
    warning_filt = noisy_stats["overlapping_frac"] < min_overlap_warn
    invalid_dataset = (noisy_stats["overlapping_frac"] < min_overlap).astype(bool).sum()
    for (dataset_id, primary_nodes), row in noisy_stats[warning_filt].iterrows():
        vprint(
            ("{:3s} low noisy overlap: nodes={:2d} jobid={:s} " +
            "overlap={:5.1f}%/{:2.0f}s nonoverlap={:2.0f}s " +
            "total={:2.0f}s").format(
                "!!!" if row["overlapping_frac"] < min_overlap else "***",
                primary_nodes,
                dataset_id,
                row["overlapping_frac"] * 100.0,
                row["overlapping_secs"],
                row["non-overlapping_secs"],
                row["total_secs"]))
    if invalid_dataset:
        raise JobOverlapError("insufficient noisy overlap")

def pivot_to_losses(dataframe, index_on=None):
    """Generates a dataframe of losses
    
    Args:
        dataframe (pandas.DataFrame): Output of load_contention_datasets
        index_on (list or None): Index on listed columns in addition to
            dataset_id and primary_nodes (default: "workload")

    Returns:
        pandas.DataFrame
    """
    if index_on is None:
        pivoted_df = dataframe.pivot_table(
            index=["dataset_id", "workload", "primary_nodes"],
            values=["performance"],
            columns=["contention"])
    elif isinstance(index_on, list):
        pivoted_df = dataframe.pivot_table(
            index=["dataset_id"] + index_on + ["primary_nodes"],
            values=["performance"],
            columns=["contention"])
    elif isinstance(index_on, str):
        pivoted_df = dataframe.pivot_table(
            index=["dataset_id"] + [index_on] + ["primary_nodes"],
            values=["performance"],
            columns=["contention"])
    else:
        raise ValueError("index_on is invalid type {}".format(type(index_on)))

    # set columns to be quiet or noisy and get rid of other multilevel columns
    pivoted_df.columns = pivoted_df.columns.get_level_values(1)

    pivoted_df["loss"] = pivoted_df["quiet"] - pivoted_df["noisy"]
    pivoted_df["loss%"] = pivoted_df["loss"] / pivoted_df["quiet"]

    return pivoted_df

def pivot_and_subselect_workload(dataframe, workload, workload_col, perf_key):
    # create loss dataframe
    pivoted_df = pivot_to_losses(dataframe, index_on=workload_col)
    
    # subselect the correct data to be bucketed
    if isinstance(workload_col, str) or len(workload_col) == 1:
        if workload not in pivoted_df.index.get_level_values(1):
            return None
        plot_series = pivoted_df.loc[:, workload, :].reset_index()[["primary_nodes", perf_key]]
        access = ""
        metric = workload
    elif not isinstance(workload_col):
        raise ValueError("workload_col must be str or list")
    elif len(workload_col) == 2:
        plot_series = pivoted_df.loc[:, :, workload, :].reset_index()[["primary_nodes", perf_key]]
        access, metric = workload.split(None, 1)
        access += " "
    elif len(workload_col) == 3:
        plot_series = pivoted_df.loc[:, :, :, workload, :].reset_index()[["primary_nodes", perf_key]]
        access = ""
        metric = workload
    else:
        raise NotImplemented("workload_col only supported up to 3 columns")

    return plot_series
