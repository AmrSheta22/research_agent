import pandas as pd
from . import utils_case_control
from . import utils_cross_sectional
from . import utils_controlled_before_after
from . import utils_crossover_RCT
from . import utils_individual_RCT
from . import utils_pre_post
from . import utils_retrospective
# =============================================================================
# Cross Sectional Study Design
# =============================================================================

def conduct_cross_sectional_cont_iv_multinomial_dv(df, iv, dv, covariates=None):
    return utils_cross_sectional.test_cont_indep_multinomial_dep(df, iv, dv, covariates)


def conduct_cross_sectional_cont_iv_cont_dv(df, iv, dv, covariates=None):
    return utils_cross_sectional.test_cont_indep_cont_dep(df, iv, dv, covariates)


def conduct_cross_sectional_cont_iv_binary_dv(df, iv, dv, covariates=None):
    return utils_cross_sectional.test_cont_indep_binary_dep(df, iv, dv, covariates)


def conduct_cross_sectional_binary_iv_binary_dv(df, iv, dv, covariates=None):
    return utils_cross_sectional.test_binary_indep_binary_dep(df, iv, dv, covariates)


def conduct_cross_sectional_binary_iv_cont_dv(df, iv, dv, covariates=None):
    return utils_cross_sectional.test_binary_indep_cont_dep(df, iv, dv, covariates)


def conduct_cross_sectional_binary_iv_multinomial_dv(df, iv, dv, covariates=None):
    return utils_cross_sectional.test_binary_indep_multinomial_dep(
        df, iv, dv, covariates
    )


def conduct_cross_sectional_multinomial_iv_binary_dv(df, iv, dv, covariates=None):
    return utils_cross_sectional.test_multinomial_indep_binary_dep(
        df, iv, dv, covariates
    )


def conduct_cross_sectional_multinomial_iv_cont_dv(df, iv, dv, covariates=None):
    return utils_cross_sectional.test_multinomial_indep_cont_dep(df, iv, dv, covariates)


def conduct_cross_sectional_multinomial_iv_multinomial_dv(df, iv, dv, covariates=None):
    return utils_cross_sectional.test_multinomial_indep_multinomial_dep(
        df, iv, dv, covariates
    )


# =============================================================================
# Case Control Study Design
# =============================================================================


def conduct_case_control_binary_dv(df, case_control_col, dv, covariates=None):
    # Note: case_control_col implies the group assignment (Case vs Control) which is often treated as the main variable
    return utils_case_control.case_control_binary_dependent_variable_test(
        df, case_control_col, dv, covariates
    )


def conduct_case_control_multinomial_dv(df, case_control_col, dv, covariates=None):
    return utils_case_control.case_control_multinomial_dependent_variable_test(
        df, case_control_col, dv, covariates
    )


def conduct_case_control_cont_dv(df, case_control_col, dv, covariates=None):
    return utils_case_control.case_control_continous_dependent_variable_test(
        df, case_control_col, dv, covariates
    )


# =============================================================================
# Controlled Before-After Study Design
# =============================================================================


def conduct_controlled_before_after_binary_iv_cont_dv(
    df, dv, treatment, time, covariates=None, subject_id=None
):
    return (
        utils_controlled_before_after.controlled_before_after_binary_iv_continuous_dv(
            df, dv, treatment, time, covariates, subject_id
        )
    )


def conduct_controlled_before_after_binary_iv_binary_dv(
    df, dv, treatment, time, covariates=None, subject_id=None
):
    return utils_controlled_before_after.controlled_before_after_binary_iv_binary_dv_with_time(
        df, dv, treatment, time, covariates, subject_id
    )


def conduct_controlled_before_after_binary_iv_multinomial_dv(
    df, dv, treatment, time, covariates=None, subject_id=None
):
    return utils_controlled_before_after.controlled_before_after_binary_iv_multinomial_dv_with_time(
        df, dv, treatment, time, covariates, subject_id
    )


# =============================================================================
# Crossover RCT Study Design
# =============================================================================
# Note: utils_crossover_RCT expect specific column names 'treatment', 'outcome', 'subject_id'
# We rename columns in a copy of the dataframe to match.


def _prepare_crossover_df(df, iv, dv, subject_id):
    df_copy = df.copy()
    # If the user provides columns that are already named treatment/outcome/subject_id but mapped differently,
    # we need to be careful. But generally standardizing is safer.
    rename_map = {iv: "treatment", dv: "outcome", subject_id: "subject_id"}
    return df_copy.rename(columns=rename_map)


def conduct_crossover_rct_binary_iv_cont_dv(df, iv, dv, subject_id):
    df_c = _prepare_crossover_df(df, iv, dv, subject_id)
    return utils_crossover_RCT.case_1_binaryIV_contDV(df_c)


def conduct_crossover_rct_binary_iv_binary_dv(df, iv, dv, subject_id):
    df_c = _prepare_crossover_df(df, iv, dv, subject_id)
    return utils_crossover_RCT.case_2_binaryIV_binaryDV(df_c)


def conduct_crossover_rct_binary_iv_multinomial_dv(df, iv, dv, subject_id):
    df_c = _prepare_crossover_df(df, iv, dv, subject_id)
    return utils_crossover_RCT.case_3_binaryIV_multinomialDV(df_c)


def conduct_crossover_rct_categorical_iv_cont_dv(df, iv, dv, subject_id):
    # Case 4
    df_c = _prepare_crossover_df(df, iv, dv, subject_id)
    return utils_crossover_RCT.case_4_categoricalIV_contDV(df_c)


def conduct_crossover_rct_multinomial_iv_binary_dv(df, iv, dv, subject_id):
    # Case 5
    df_c = _prepare_crossover_df(df, iv, dv, subject_id)
    return utils_crossover_RCT.case_5_multinomialIV_binaryDV(df_c)


def conduct_crossover_rct_multinomial_iv_multinomial_dv(df, iv, dv, subject_id):
    # Case 6
    df_c = _prepare_crossover_df(df, iv, dv, subject_id)
    return utils_crossover_RCT.case_6_multinomialIV_multinomialDV(df_c)


def conduct_crossover_rct_ordinal_iv_cont_dv(df, iv, dv, subject_id):
    # Case 7
    df_c = _prepare_crossover_df(df, iv, dv, subject_id)
    return utils_crossover_RCT.case_7_ordinalIV_contDV(df_c)


def conduct_crossover_rct_ordinal_iv_binary_dv(df, iv, dv, subject_id):
    # Case 8
    df_c = _prepare_crossover_df(df, iv, dv, subject_id)
    return utils_crossover_RCT.case_8_ordinalIV_binaryDV(df_c)


def conduct_crossover_rct_ordinal_iv_multinomial_dv(df, iv, dv, subject_id):
    # Case 9
    df_c = _prepare_crossover_df(df, iv, dv, subject_id)
    return utils_crossover_RCT.case_9_ordinalIV_multinomialDV(df_c)


# =============================================================================
# Individual RCT Study Design
# =============================================================================


def conduct_individual_rct_binary_iv_cont_dv(df, iv, dv, covariates=None):
    return utils_individual_RCT.run_all_tests_binaryIV_continuousDV(
        df, iv, dv, covariates
    )


def conduct_individual_rct_binary_iv_binary_dv(df, iv, dv, covariates=None):
    return utils_individual_RCT.run_all_tests_binaryIV_binaryDV(df, iv, dv, covariates)


def conduct_individual_rct_binary_iv_multinomial_dv(df, iv, dv, covariates=None):
    return utils_individual_RCT.run_all_tests_binaryIV_multinomialDV(
        df, iv, dv, covariates
    )


def conduct_individual_rct_multinomial_iv_cont_dv(df, iv, dv, covariates=None):
    return utils_individual_RCT.run_all_tests_multinomialIV_continuousDV(
        df, iv, dv, covariates
    )


def conduct_individual_rct_multinomial_iv_binary_dv(df, iv, dv, covariates=None):
    return utils_individual_RCT.run_all_tests_multinomialIV_binaryDV(
        df, iv, dv, covariates
    )


def conduct_individual_rct_multinomial_iv_multinomial_dv(df, iv, dv, covariates=None):
    return utils_individual_RCT.run_all_tests_multinomialIV_multinomialDV(
        df, iv, dv, covariates
    )


def conduct_individual_rct_ordinal_iv_cont_dv(df, iv, dv, covariates=None):
    return utils_individual_RCT.run_all_tests_ordinalIV_continuousDV(
        df, iv, dv, covariates
    )


def conduct_individual_rct_ordinal_iv_binary_dv(df, iv, dv, covariates=None):
    return utils_individual_RCT.run_all_tests_ordinalIV_binaryDV(df, iv, dv, covariates)


def conduct_individual_rct_ordinal_iv_multinomial_dv(df, iv, dv, covariates=None):
    return utils_individual_RCT.run_all_tests_ordinalIV_categoricalDV(
        df, iv, dv, covariates
    )


# =============================================================================
# Pre-Post Study Design
# =============================================================================


def conduct_pre_post_cont_dv(df, pre, post, covariates=None):
    return utils_pre_post.run_all_tests_continuous_prepost(df, pre, post, covariates)


def conduct_pre_post_binary_dv(df, pre, post, covariates=None):
    return utils_pre_post.run_all_tests_binary_prepost(df, pre, post, covariates)


def conduct_pre_post_multinomial_dv(df, pre, post, covariates=None):
    return utils_pre_post.run_all_tests_multinomial_prepost(df, pre, post, covariates)


def conduct_pre_post_ordinal_dv(df, pre, post, covariates=None):
    return utils_pre_post.run_all_tests_ordinal_prepost(df, pre, post, covariates)


# =============================================================================
# Retrospective Study Design
# =============================================================================


def conduct_retrospective_cont_iv_cont_dv(df, iv, dv, covariates=None):
    return utils_retrospective.run_all_tests_continuousIV_continuousDV(
        df, iv, dv, covariates
    )


def conduct_retrospective_cont_iv_binary_dv(df, iv, dv, covariates=None):
    return utils_retrospective.run_all_tests_continuousIV_binaryDV(
        df, iv, dv, covariates
    )


def conduct_retrospective_cont_iv_multinomial_dv(df, iv, dv, covariates=None):
    return utils_retrospective.run_all_tests_continuousIV_multinomialDV(
        df, iv, dv, covariates
    )


def conduct_retrospective_binary_iv_cont_dv(df, iv, dv, covariates=None):
    return utils_retrospective.run_all_tests_binaryIV_continuousDV(
        df, iv, dv, covariates
    )


def conduct_retrospective_binary_iv_binary_dv(df, iv, dv, covariates=None):
    return utils_retrospective.run_all_tests_binaryIV_binaryDV(df, iv, dv, covariates)


def conduct_retrospective_binary_iv_multinomial_dv(df, iv, dv, covariates=None):
    return utils_retrospective.run_all_tests_binaryIV_multinomialDV(
        df, iv, dv, covariates
    )


def conduct_retrospective_multinomial_iv_cont_dv(df, iv, dv, covariates=None):
    return utils_retrospective.run_all_tests_multinomialIV_continuousDV(
        df, iv, dv, covariates
    )


def conduct_retrospective_multinomial_iv_binary_dv(df, iv, dv, covariates=None):
    return utils_retrospective.run_all_tests_multinomialIV_binaryDV(
        df, iv, dv, covariates
    )


def conduct_retrospective_multinomial_iv_multinomial_dv(df, iv, dv, covariates=None):
    return utils_retrospective.run_all_tests_multinomialIV_multinomialDV(
        df, iv, dv, covariates
    )
