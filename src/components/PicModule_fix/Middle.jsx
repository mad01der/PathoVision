/*
 * @Date: 2022-07-21 10:17:01
 * @LastEditors: JZY
 * @LastEditTime: 2022-10-04 11:40:28
 * @FilePath: /visual/src/components/CoreModule/ScatterModel/index.jsx
 */

import React, { Component } from "react";
import { Row, Col } from "antd";
import Pic from "./Pic";
export default class Middle extends Component {
  constructor(props) {
    super(props);
    this.state = {
      patcheId: -1,
      imgId: -1,
      selectedPatch: -1,
      loading: true,
      choosePatches: props.choosePatches,
    };
  }
  changeDeletePatches = (p) => {
    this.setState({
      choosePatches: p,
    });
    this.picChildRef.changeDeletePatches(p);
  };

  handlePicChildEvent = (ref) => {
    this.picChildRef = ref;
  };
  changeChoosePatches = async (p) => {
    this.setState({
      choosePatches: p,
    });
    this.props.changeChoosePatches(p);
  };
  handlePicChildEvent = (ref) => {
    this.picChildRef = ref;
  };
  render() {
    return (
      <>
        <Row gutter={5}>
          <Col span={ 24}>
              <Pic
                changeChoosePatches={this.changeChoosePatches}
                choosePatches={this.state.choosePatches}
                onChildEvent={this.handlePicChildEvent}
                WSI_Data={this.props.WSI_Data}
                ref={this.picChildRef}
              />
          </Col>
        </Row>
      </>
    );
  }
}
