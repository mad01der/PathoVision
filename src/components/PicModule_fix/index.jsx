/*
 * @Date: 2022-04-17 18:26:37
 * @LastEditors: JZY
 * @LastEditTime: 2022-12-17 17:22:12
 * @FilePath: /project/Visual/src/components/CoreModule/index.jsx
 */
import React, { Component } from "react";
import { Row, Col, Card } from "antd";
import Middle from "./Middle";

export default class PicModule extends Component {
  constructor(props) {
    super(props);
    this.state = {
      choosePatches: props.choosePatches,
    };
  }
  componentDidMount = () => {
    this.props.onChildEvent(this);
  };
  changeDeletePatches = (p) => {
    this.setState({
      choosePatches: p,
    });
    this.middleModelRef.changeDeletePatches(p);
  };
  changeChoosePatches = async (p) => {
    await this.setState({
      choosePatches: p,
    });
    this.props.changeChoosePatches(this.state.choosePatches);
  };
  handlePicModuleEvent = (ref) => {
    this.middleModelRef = ref;
  };
  render() {
    return (
      <>
        <Card bordered={false} hoverable={true}>
          <Row>
            <Col span={24}>
              <Middle
                onChildEvent={this.handlePicModuleEvent}
                ref={this.middleModelRef}
                changeChoosePatches={this.changeChoosePatches}
                choosePatches={this.state.choosePatches}
                sample_Data={this.props.sample_Data}
                WSI_Data={this.props.WSI_Data}
              />
            </Col>
          </Row>
        </Card>
      </>
    );
  }
}
