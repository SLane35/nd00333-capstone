from sklearn.linear_model import LinearRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset, Datastore
from azureml.data.datapath import DataPath
from azureml.core import Workspace

ws = Workspace.from_config()

datastore_name='workspaceblobstore'
datastore=Datastore.get(ws,datastore_name)

datastore_path = [(datastore, 'UI/01-03-2021_103243_UTC/LaborPredictionData6.csv')]
ds = Dataset.Tabular.from_delimited_files(path=datastore_path)
ds = ds.take(3000).to_pandas_dataframe()

def clean_data(dataset):
    dataset.itemset = dataset.itemset.fillna('')

    dataset.loc[dataset.itemset == 'wrap around rubber', 'itemset'] = 'Wrap Around Rubber'
    dataset.loc[dataset.itemset == 'putty', 'itemset'] = 'PUTTY'
    dataset.loc[dataset.itemset == 'Putty', 'itemset'] = 'PUTTY'
    dataset.loc[dataset.itemset == 'Putty Hack Glaze', 'itemset'] = 'PUTTY'
    dataset.loc[dataset.itemset == 'Wood Beads', 'itemset'] = 'WOOD BEADS'
    dataset.loc[dataset.itemset == 'Pressure Plate', 'itemset'] = 'PRESSURE PLATE'
    dataset.loc[dataset.itemset == 'Snap Beads', 'itemset'] = 'SNAP BEADS'
    dataset.loc[dataset.itemset == 'Wrap Around Rubber', 'itemset'] = 'WRAP AROUND RUBBER'
    dataset.loc[dataset.itemset == 'Wrap Around Frame', 'itemset'] = 'WRAP AROUND FRAME'
    dataset.loc[dataset.itemset == 'metal', 'itemset'] = 'STEEL BEADS'
    dataset.loc[dataset.itemset == 'Silicone', 'itemset'] = 'VINYL BEADS'

    dataset.loc[dataset.itemset == 'Aluminum Beads', 'itemset'] = 'OTHER'
    dataset.loc[dataset.itemset == 'NO BEADS', 'itemset'] = 'OTHER'
    dataset.loc[dataset.itemset == 'PUSH RUBBER', 'itemset'] = 'OTHER'
    dataset.loc[dataset.itemset == 'STEEL BEADS', 'itemset'] = 'OTHER'
    dataset.loc[dataset.itemset == 'ZIPPER WALL', 'itemset'] = 'OTHER'
    
    dataset = pd.concat([dataset, pd.get_dummies(dataset['itemset'], prefix='itemset')], axis=1)
    dataset = dataset.drop('itemset', axis=1)
    
    dataset.loc[dataset.leadmechanic == 67, 'leadmechanic'] = 'OTHER'
    dataset.loc[dataset.leadmechanic == 69, 'leadmechanic'] = 'OTHER'
    dataset.loc[dataset.leadmechanic == 74, 'leadmechanic'] = 'OTHER'
    dataset.loc[dataset.leadmechanic == 138, 'leadmechanic'] = 'OTHER'
    dataset.loc[dataset.leadmechanic == 269, 'leadmechanic'] = 'OTHER'
    dataset.loc[dataset.leadmechanic == 132, 'leadmechanic'] = 'OTHER'
    dataset.loc[dataset.leadmechanic == 139, 'leadmechanic'] = 'OTHER'
    dataset.loc[dataset.leadmechanic == 137, 'leadmechanic'] = 'OTHER'
    
    dataset = pd.concat([dataset, pd.get_dummies(dataset['leadmechanic'], prefix='leadmechanic')], axis=1)
    dataset = dataset.drop('leadmechanic', axis=1)
    
    dataset = pd.concat([dataset, pd.get_dummies(dataset['locationtype'], prefix='locationtype')], axis=1)
    dataset = dataset.drop('locationtype', axis=1)
    
    dataset.framemat = dataset.framemat.fillna('')

    dataset.loc[dataset.framemat == 'wood', 'framemat'] = 'WOOD'
    dataset.loc[dataset.framemat == 'Wood', 'framemat'] = 'WOOD'
    dataset.loc[dataset.framemat == 'Aluminum', 'framemat'] = 'ALUMINUM'
    dataset.loc[dataset.framemat == 'alum', 'framemat'] = 'ALUMINUM'
    dataset.loc[dataset.framemat == '450', 'framemat'] = 'ALUMINUM'
    dataset.loc[dataset.framemat == '451', 'framemat'] = 'ALUMINUM'
    dataset.loc[dataset.framemat == 'Clear Alum', 'framemat'] = 'ALUMINUM'
    dataset.loc[dataset.framemat == 'Metal', 'framemat'] = 'ALUMINUM'
    dataset.loc[dataset.framemat == 'METAL', 'framemat'] = 'ALUMINUM'
    dataset.loc[dataset.framemat == 'STEEL', 'framemat'] = 'ALUMINUM'
    dataset.loc[dataset.framemat == 'STAINLESS STEEL', 'framemat'] = 'ALUMINUM'
    dataset.loc[dataset.framemat == 'ALUMIUM', 'framemat'] = 'ALUMINUM'
    dataset.loc[dataset.framemat == 'STEEL SASH', 'framemat'] = 'ALUMINUM'
    dataset.loc[dataset.framemat == 'Vinyl', 'framemat'] = 'VINYL'
    dataset.loc[dataset.framemat == 'VINYL SH', 'framemat'] = 'VINYL'


    dataset.loc[dataset.framemat == 'BOTTOM SH', 'framemat'] = 'OTHER'
    dataset.loc[dataset.framemat == 'DIVISION BAR', 'framemat'] = 'OTHER'
    dataset.loc[dataset.framemat == 'FINTUBE', 'framemat'] = 'OTHER'
    dataset.loc[dataset.framemat == 'PELLA', 'framemat'] = 'OTHER'
    
    dataset = pd.concat([dataset, pd.get_dummies(dataset['framemat'], prefix='framemat')], axis=1)
    dataset = dataset.drop('framemat', axis=1)
    
    dataset.loc[dataset.framefunc == 'Bottom of DH', 'framefunc'] = 'BOTTOM DH'
    dataset.loc[dataset.framefunc == 'Bottom of Double Hung', 'framefunc'] = 'BOTTOM DH'
    dataset.loc[dataset.framefunc == 'Bottom of SH', 'framefunc'] = 'BOTTOM SH'
    dataset.loc[dataset.framefunc == 'Curtainwall', 'framefunc'] = 'CURTAIN WALL'
    dataset.loc[dataset.framefunc == 'Fixed', 'framefunc'] = 'FIXED'
    dataset.loc[dataset.framefunc == 'Flush Glaze', 'framefunc'] = 'FLUSH GLAZE'
    dataset.loc[dataset.framefunc == 'GARAGE DOOR VISION PANEL', 'framefunc'] = 'GARAGE DOOR'
    dataset.loc[dataset.framefunc == 'LUG SASH TOP', 'framefunc'] = 'SASH'
    dataset.loc[dataset.framefunc == 'LUG SASH', 'framefunc'] = 'SASH'
    dataset.loc[dataset.framefunc == 'SASH RAIL', 'framefunc'] = 'SASH'
    dataset.loc[dataset.framefunc == 'TOP SASH', 'framefunc'] = 'SASH'
    dataset.loc[dataset.framefunc == 'BOTTOM SASH', 'framefunc'] = 'SASH'
    dataset.loc[dataset.framefunc == 'Transom', 'framefunc'] = 'TRANSOM'
    dataset.loc[dataset.framefunc == 'UPPER HALF OF DOUBLE HUNG', 'framefunc'] = 'TOP DH'
    dataset.loc[dataset.framefunc == 'UPPER HALF', 'framefunc'] = 'TOP SH'
    dataset.loc[dataset.framefunc == 'STATIONARY', 'framefunc'] = 'FIXED'
    dataset.loc[dataset.framefunc == 'UPPER DH', 'framefunc'] = 'TOP DH'
    dataset.loc[dataset.framefunc == 'UPPER SH', 'framefunc'] = 'TOP SH'
    dataset.loc[dataset.framefunc == 'Slider', 'framefunc'] = 'SLIDER'
    dataset.loc[dataset.framefunc == 'Stat Window', 'framefunc'] = 'FIXED'

    dataset.loc[dataset.framefunc == '451', 'framefunc'] = 'OTHER'
    dataset.loc[dataset.framefunc == 'Door', 'framefunc'] = 'OTHER'
    dataset.loc[dataset.framefunc == 'FINTUBE SIDELITE', 'framefunc'] = 'OTHER'
    dataset.loc[dataset.framefunc == 'GARAGE DOOR', 'framefunc'] = 'OTHER'
    dataset.loc[dataset.framefunc == 'HOPPER WINDOW', 'framefunc'] = 'OTHER'
    dataset.loc[dataset.framefunc == 'Hand Rail', 'framefunc'] = 'OTHER'
    dataset.loc[dataset.framefunc == 'PICTURE WINDOW', 'framefunc'] = 'OTHER'
    dataset.loc[dataset.framefunc == 'SKYLITE', 'framefunc'] = 'OTHER'
    dataset.loc[dataset.framefunc == 'STORM DOOR', 'framefunc'] = 'OTHER'
    
    dataset = pd.concat([dataset, pd.get_dummies(dataset['framefunc'], prefix='framefunc')], axis=1)
    dataset = dataset.drop('framefunc', axis=1)
    
    dataset.LongCarry = dataset.LongCarry.fillna(0)
    
    dataset = dataset.astype({"itemset_": bool,"itemset_OTHER": bool,"itemset_PRESSURE PLATE": bool,"itemset_PUTTY": bool, "itemset_SCREW BEADS": bool, "itemset_SNAP BEADS": bool, "itemset_SNAP BEADS": bool,"itemset_VINYL BEADS": bool, "itemset_WOOD BEADS": bool,"itemset_WRAP AROUND FRAME": bool,"itemset_WRAP AROUND RUBBER": bool,  "itemset_flush glaze": bool, "leadmechanic_6":bool, "leadmechanic_11":bool, "leadmechanic_62":bool,"leadmechanic_90":bool,"leadmechanic_118":bool, "leadmechanic_211":bool, "leadmechanic_228":bool, "leadmechanic_OTHER":bool, "locationtype_0": bool, "locationtype_1": bool, "locationtype_2": bool, "locationtype_4": bool, "locationtype_5": bool, "locationtype_6": bool, "locationtype_7": bool, "locationtype_9": bool, "locationtype_10": bool, "locationtype_11": bool, "locationtype_12": bool, "locationtype_13": bool, "framemat_": bool, "framemat_ALUMINUM": bool, "framemat_ANDERSEN":bool, "framemat_SASH RAIL": bool, "framemat_VINYL": bool, "framemat_WOOD":bool, "framemat_OTHER":bool, "framefunc_BOTTOM DH": bool, "framefunc_BOTTOM SH": bool, "framefunc_CASEMENT": bool, "framefunc_CURTAIN WALL":bool, "framefunc_FIXED": bool, "framefunc_FLUSH GLAZE": bool, "framefunc_OTHER": bool, "framefunc_SASH":bool, "framefunc_SIDELITE":bool, "framefunc_SLIDER":bool, "framefunc_TOP DH": bool, "framefunc_TOP SH": bool, "framefunc_TRANSOM": bool, "LongCarry": bool})
    
    heightmax = dataset['heightwholenum'].max()
    widthmax = dataset['widthWholeNum'].max()
    liftmax = dataset['lift'].max()
    
    dataset['heightwholenum'] = dataset['heightwholenum']/heightmax
    dataset['widthWholeNum'] = dataset['widthWholeNum']/widthmax
    dataset['lift'] = dataset['lift']/liftmax
    
    x = dataset[['heightwholenum','widthWholeNum','lift','LongCarry', 'itemset_','itemset_OTHER', 'itemset_PRESSURE PLATE','itemset_PUTTY','itemset_SCREW BEADS','itemset_SNAP BEADS','itemset_VINYL BEADS', 'itemset_WOOD BEADS','itemset_WRAP AROUND FRAME','itemset_WRAP AROUND RUBBER','itemset_flush glaze', 'leadmechanic_6','leadmechanic_11','leadmechanic_62','leadmechanic_90','leadmechanic_118','leadmechanic_211', 'leadmechanic_228','leadmechanic_OTHER','locationtype_0','locationtype_1','locationtype_2','locationtype_4','locationtype_5','locationtype_6','locationtype_7','locationtype_9','locationtype_10','locationtype_11','locationtype_12', 'locationtype_13','framemat_','framemat_ALUMINUM','framemat_ANDERSEN','framemat_OTHER','framemat_SASH RAIL','framemat_VINYL','framemat_WOOD','framefunc_BOTTOM DH','framefunc_BOTTOM SH', 'framefunc_CASEMENT', 'framefunc_CURTAIN WALL', 'framefunc_FIXED','framefunc_FLUSH GLAZE', 'framefunc_OTHER', 'framefunc_SASH', 'framefunc_SIDELITE', 'framefunc_SLIDER','framefunc_TOP DH', 'framefunc_TOP SH','framefunc_TRANSOM']].values
    
    y = dataset['TotalInstallationTime'].values

    return x, y

x, y = clean_data(ds)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8)

train = np.concatenate((x_train,y_train), axis=1)

run = Run.get_context()
    

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--fit_intercept', type=bool, default=True)
    parser.add_argument('--normalize', type=bool, default=False)

    args = parser.parse_args()

    run.log("Fit Intercept:", np.bool(args.fit_intercept))
    run.log("Normalize:", np.bool(args.normalize))

    model = LinearRegression(fit_intercept=args.fit_intercept, normalize=args.normalize).fit(x_train, y_train)

    rmse = model.score(x_test, y_test, squared=True)
    run.log("RMSE", np.float(rmse))
    

if __name__ == '__main__':
    main()
